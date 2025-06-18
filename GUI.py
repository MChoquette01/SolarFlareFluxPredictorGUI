import PySimpleGUI as psg  # https://pypi.org/project/PySimpleGUI-4-foss/
import App
import os
import threading
import time
from datetime import datetime, timezone
import matplotlib
import textwrap
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


# See here: https://github.com/PySimpleGUI/PySimpleGUI/blob/master/DemoPrograms/Demo_Matplotlib_Embedded_Toolbar.py
def draw_figure_w_toolbar(canvas, fig, canvas_toolbar):
    if canvas.children:
        for child in canvas.winfo_children():
            child.destroy()
    if canvas_toolbar.children:
        for child in canvas_toolbar.winfo_children():
            child.destroy()
    figure_canvas_agg = FigureCanvasTkAgg(fig, master=canvas)
    figure_canvas_agg.draw()
    toolbar = Toolbar(figure_canvas_agg, canvas_toolbar)
    toolbar.update()
    figure_canvas_agg.get_tk_widget().pack(side='right', fill='both', expand=1)

# used for adding matplotlib widget to graph
class Toolbar(NavigationToolbar2Tk):
    def __init__(self, *args, **kwargs):
        super(Toolbar, self).__init__(*args, **kwargs)


def interactive_graph_update():
    """Update the graph of XRSA/B flux"""
    plt.figure(figsize=(15, 3))
    plt.semilogy(l.timestamps[::-1], l.input_data["XRSB_flux"][:len(l.timestamps)][::-1], color='blue', label="XRSB")
    plt.semilogy(l.timestamps[::-1], l.input_data["XRSA_flux"][:len(l.timestamps)][::-1], color='orange', label="XRSA")
    x_labels = [x[-9:-4] for x in l.timestamps[::-1]][::3]
    plt.xticks(ticks=[x for x in range(0, len(l.timestamps), 3)], labels=x_labels, rotation=45)
    plt.yticks(ticks=[10 ** -8, 10 ** -7, 10 ** -6, 5 * 10 ** -6, 10 ** -5, 10 ** -4],
               labels=["A", "B", "C", "C5", "M", "X"])
    plt.xlabel("UTC Time (HH:MM)")
    plt.ylabel("Flux Level")
    plt.legend()
    plt.tight_layout()
    # plt.figure(1)
    fig = plt.gcf()
    DPI = fig.get_dpi()
    # ------------------------------- you have to play with this size to reduce the movement error when the mouse hovers over the figure, it's close to canvas size
    fig.set_size_inches(1600 / float(DPI), 250 / float(DPI))
    # ------------------------------- you have to play with this size to reduce the movement error when the mouse hovers over the figure, it's close to canvas size

    draw_figure_w_toolbar(window['canvas'].TKCanvas, fig, window['controls_cv'].TKCanvas)


def fig_maker(window):  # this should be called as a thread, then time.sleep() here would not freeze the GUI
    plt.figure(figsize=(15, 3))
    plt.semilogy(l.timestamps[::-1], l.input_data["XRSB_flux"][:len(l.timestamps)][::-1], color='blue', label="XRSB")
    plt.semilogy(l.timestamps[::-1], l.input_data["XRSA_flux"][:len(l.timestamps)][::-1], color='orange', label="XRSA")
    x_labels = [x[-9:-4] for x in l.timestamps[::-1]][::3]
    plt.xticks(ticks=[x for x in range(0, len(l.timestamps), 3)], labels=x_labels, rotation=45)
    plt.yticks(ticks=[10**-8, 10**-7, 10**-6, 5*10**-6, 10**-5, 10**-4], labels=["A", "B", "C", "C5", "M", "X"])
    plt.xlabel("UTC Time (HH:MM)")
    plt.ylabel("$W m^{-2}$")
    plt.legend()
    plt.tight_layout()
    window.write_event_value('-THREAD-', 'done.')
    # time.sleep(1)
    return plt.gcf()


def draw_figure(canvas, figure, loc=(0, 0)):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def delete_fig_agg(fig_agg):
    fig_agg.get_tk_widget().forget()
    plt.close('all')


def format_prediction_string(stop_minute=15):

    formatted_pred_string = ""
    for minutes_since_start in range(-5, 16):
        if l.prediction_history[minutes_since_start]["<C5"] is not None:
            if l.in_flare_mode and minutes_since_start >= 1:
                time_since_flare_start_total_seconds = (l.real_data_timestamps[l.minutes_since_start - minutes_since_start] - l.flare_start_time).total_seconds()
                time_since_flare_start_minutes = int(time_since_flare_start_total_seconds / 60)
                time_since_flare_start_seconds = time_since_flare_start_total_seconds % 60
                delay_since_flare_start = f"(+ {time_since_flare_start_minutes:.2f}:{time_since_flare_start_seconds:.2f})"
            else:
                delay_since_flare_start = ""
            if l.prediction_history[minutes_since_start]["<C5"] >= 0.5:
                formatted_pred_string += f"{minutes_since_start}: <C5: "
            else:
                formatted_pred_string += f"{minutes_since_start}: >=C5: "
            for target in l.targets:
                formatted_pred_string += f" {target}: {l.prediction_history[minutes_since_start][target]:.3f}"
            formatted_pred_string += delay_since_flare_start
            formatted_pred_string += "\n"
            if minutes_since_start == stop_minute:
                return formatted_pred_string
        else:
            print(minutes_since_start)
            print(l.prediction_history)
            return formatted_pred_string  # return whatever we've written
    return formatted_pred_string


def update_current_table():
    """Rewrite table of live predictions"""
    table_formatted_preds = []
    for minutes_since_start in range(l.minutes_since_start, -6, -1):
        if l.prediction_history[minutes_since_start]["<C5"] is not None:  # if one is None they'll all be None at a time
            if l.in_flare_mode and minutes_since_start >= 1:
                time_since_flare_start_total_seconds = (l.real_data_timestamps[l.minutes_since_start - minutes_since_start] - l.flare_start_time).total_seconds()
                time_since_flare_start_minutes = int(time_since_flare_start_total_seconds / 60)
                time_since_flare_start_seconds = time_since_flare_start_total_seconds % 60
                delay_since_flare_start = f"{time_since_flare_start_minutes:.0f}:{time_since_flare_start_seconds:.2f}"
            else:
                delay_since_flare_start = ""
            prediction = "<C5" if l.prediction_history[minutes_since_start]["<C5"] >= 0.5 else ">=C5"
            table_formatted_preds.append([minutes_since_start,
                                          prediction,
                                          round(l.prediction_history[minutes_since_start]["<C5"], 2),
                                          round(l.prediction_history[minutes_since_start]["C, >=C5"], 2),
                                          round(l.prediction_history[minutes_since_start]["M"], 2),
                                          round(l.prediction_history[minutes_since_start]["X"], 2),
                                          delay_since_flare_start])
    window["CurrentPredictionTable"].update(table_formatted_preds)


def update_latest_flare_table():
    """Rewrite table of latest full flare predictions"""
    table_formatted_preds = []
    for minutes_since_start in range(l.minutes_since_start, -6, -1):
        if l.prediction_history[minutes_since_start]["<C5"] is not None:  # if one is None they'll all be None at a time
            if l.in_flare_mode and minutes_since_start >= 1:
                time_since_flare_start_total_seconds = (l.real_data_timestamps[l.minutes_since_start - minutes_since_start] - l.flare_start_time).total_seconds()
                time_since_flare_start_minutes = int(time_since_flare_start_total_seconds / 60)
                time_since_flare_start_seconds = time_since_flare_start_total_seconds % 60
                delay_since_flare_start = f"{time_since_flare_start_minutes:.0f}:{time_since_flare_start_seconds:.2f}"
            else:
                delay_since_flare_start = ""
            prediction = "<C5" if l.prediction_history[minutes_since_start]["<C5"] >= 0.5 else ">=C5"
            table_formatted_preds.append([minutes_since_start,
                                          prediction,
                                          round(l.prediction_history[minutes_since_start]["<C5"], 2),
                                          round(l.prediction_history[minutes_since_start]["C, >=C5"], 2),
                                          round(l.prediction_history[minutes_since_start]["M"], 2),
                                          round(l.prediction_history[minutes_since_start]["X"], 2),
                                          delay_since_flare_start])
    window["LatestPredictionTable"].update(table_formatted_preds)


def worker_thread1(fig_agg, window):
    """
    Main logic thread that communicates with the GUI
    These threads can call functions that block without affecting the GUI (a good thing)
    Note that this function is the code started as each thread. All threads are identical in this way
    """
    l.select_mode("Warmup")
    l.warmup_mode()
    update_gui(window)
    l.select_mode("Standby")
    while True:
        while l.in_standby_mode:
            # time.sleep(1)
            l.standby_mode()
            # update GUI
            interactive_graph_update()
            # if fig_agg is not None:
            #     delete_fig_agg(fig_agg)
            # fig = fig_maker(window)
            # fig_agg = draw_figure(window['canvas'].TKCanvas, fig)
            # window.Refresh()
            update_gui(window)
            update_current_table()
            # window['predictions'].update(format_prediction_string(stop_minute=l.minutes_since_start))
        while l.minutes_since_start < 15:
            l.flare_mode()
            interactive_graph_update()
            # update GUI
            # if fig_agg is not None:
            #     delete_fig_agg(fig_agg)
            # fig = fig_maker(window)
            # fig_agg = draw_figure(window['canvas'].TKCanvas, fig)
            # window.Refresh()
            update_gui(window)
            # window['predictions'].update(format_prediction_string(stop_minute=l.minutes_since_start))
            update_current_table()
        # window['most_recent_predictions'].update(format_prediction_string(stop_minute=l.minutes_since_start))
        update_latest_flare_table()
        plt.savefig(os.path.join(l.make_flare_output_folder(), f"XRS_Flux.png"))
        l.exit_flare_mode()
        l.select_mode("Standby")


def update_gui(window):
    """Redraw cosmetic stuff and labels as flares progress"""
    if l.in_startup_mode:
        window['WarmupText'].update(font=("Helvetica", 16, "bold"), text_color="green")
        window['StandbyText'].update(font=("Helvetica", 12), text_color='black')
        window['FlareText'].update(font=("Helvetica", 12), text_color='black')
    elif l.in_standby_mode:
        window['WarmupText'].update(font=("Helvetica", 12), text_color='black')
        window['StandbyText'].update(font=("Helvetica", 16, "bold"), text_color="green")
        window['FlareText'].update(font=("Helvetica", 12), text_color="black")
    elif l.in_flare_mode:
        window['WarmupText'].update(font=("Helvetica", 12), text_color='black')
        window['StandbyText'].update(font=("Helvetica", 12), text_color='black')
        window['FlareText'].update(font=("Helvetica", 16, "bold"), text_color="green")

    window["FlareModeLock"].update(f"Flare Mode Unlocked: {l.flare_mode_unlocked}")
    window["ForceStandbyMode"].update(disabled=(not l.in_flare_mode))
    window["CurrentSatellite"].update(l.satellite)
    if len(l.real_data_timestamps) >= 2:
        window["LatestDataResponseTime"].update(l.real_data_timestamps[0] - l.real_data_timestamps[1])
    if len(l.real_data_timestamps) >= 3:
        window["SecondLatestDataResponseTime"].update(l.real_data_timestamps[1] - l.real_data_timestamps[2])
    if len(l.real_data_timestamps) >= 3:
        window["ThirdLatestDataResponseTime"].update(l.real_data_timestamps[2] - l.real_data_timestamps[3])

    if l.flare_start_time is not None:
        window["FlareStartTime"].update(datetime.strftime(l.flare_start_time, "%Y-%m-%dT%H:%M:00"))

    if l.timestamps:
        window["LatestTimestamp"].update(f"{l.timestamps[0]}")

    if l.use_test_flux:
        window["TestModeText"].update("Test Mode is enabled - using test data!\nSomeone probably did this manually in App.py!")


def show_help():
    """Show help dialog"""
    print("in help!")
    with open("Help.txt", 'r') as f:
        help_text = f.read()

    wrapper = textwrap.TextWrapper(width=80, placeholder=' ...', break_long_words=False,replace_whitespace=False)
    new_text = '\n'.join(wrapper.wrap(help_text))

    layout = [[psg.Column([[psg.Text(new_text)]], scrollable=True)]]
    window = psg.Window("Help is on the way!", layout, icon='sven.ico', resizable=True, size=(600, 600), keep_on_top=True)
    while True:
        event, values = window.read()
        if event == "Exit" or event == psg.WIN_CLOSED:
            break

    window.close()

# All the stuff inside your window.
# column_1 = [[psg.Text("Predictions will be here!", key='predictions', font=("Helvetica", 14), background_color='white', text_color='black')]]
column_1 = [[psg.Text("Live Predictions", font=("Helvetica", 18, "bold"), background_color='white', text_color='black')],
            [psg.Table([[]*4],
                       key="CurrentPredictionTable",
                       headings=["Model", "Prediction", "<C5", ">=C5", "M", "X", "Duration"],
                       col_widths=[4]*7,
                       def_col_width=1,
                       cols_justification=["c", "c", "c", "c", "c", "c", "c"],
                       num_rows=15,
                       font=('Helvetica', 14))]]
# column_2 = [[psg.Text("Once a flare happens, its full prediction history will be here!", key='most_recent_predictions', font=("Helvetica", 14), background_color='white', text_color='black')]]
column_2 = [[psg.Text("Predictions for Most Recent Flare", font=("Helvetica", 18, "bold"), background_color='white', text_color='black')],
            [psg.Table([[]*3],
                       key="LatestPredictionTable",
                       headings=["Model", "Prediction", "<C5", ">=C5", "M", "X", "Duration"],
                       col_widths=[4]*7,
                       def_col_width=1,
                       cols_justification=["c", "c", "c", "c", "c", "c", "c"],
                       num_rows=15,
                       font=('Helvetica', 14))]]
column_3 = [[psg.Text("Warmup", key="WarmupText", tooltip="Initialize app and process recent data", background_color='white', text_color='black'),
             psg.Text("Standby", key="StandbyText", tooltip="Predict assuming a flare begins in 2 minutes", background_color='white', text_color='black'),
             psg.Text("Flare", key="FlareText", tooltip="A flare is happening!", background_color='white', text_color='black')],
            [psg.Text("Flare Mode Unlocked: False", key="FlareModeLock", tooltip="Wait until a flare is not occuring before going into flare mode", background_color='white', text_color='black')],
            [psg.Text("Current Satellite: ", background_color='white', text_color='black'), psg.Text("", key="CurrentSatellite", background_color='white', text_color='black')],
            [psg.HorizontalSeparator(color='blue', pad=(0, 0))],
            [psg.Text("Latest data received from (UTC):", font=("Helvetica", 10, "bold"), background_color='white', text_color='black')],
            [psg.Text("Waiting for data...", key="LatestTimestamp", background_color='white', text_color='black')],
            [psg.HorizontalSeparator(color='blue', pad=(0, 0))],
            [psg.Text("Most recent flare start time (UTC):", font=("Helvetica", 10, "bold"), background_color='white', text_color='black')],
            [psg.Text("No flares yet, check back soon!", key="FlareStartTime", background_color='white', text_color='black')],
            [psg.HorizontalSeparator(color='blue', pad=(0, 0))],
            [psg.Text("3 Last Data Delay Times", font=("Helvetica", 10, "bold"), background_color='white', text_color='black')],
            [psg.Text("1: ", background_color='white', text_color='black'), psg.Text("", key="LatestDataResponseTime", background_color='white', text_color='black'), psg.Text("s", background_color='white', text_color='black')],
            [psg.Text("2: ", background_color='white', text_color='black'), psg.Text("", key="SecondLatestDataResponseTime", background_color='white', text_color='black'), psg.Text("s", background_color='white', text_color='black')],
            [psg.Text("3: ", background_color='white', text_color='black'), psg.Text("", key="ThirdLatestDataResponseTime", background_color='white', text_color='black'), psg.Text("s", background_color='white', text_color='black')],
            [psg.HorizontalSeparator(color='blue', pad=(0, 0))],
            [psg.Button("Force Standby Mode", key="ForceStandbyMode", disabled=True)],
            [psg.Text("", key="TestModeText", background_color='white', text_color='black')],
            [psg.Button("Help", key="Help")]]

layout = [[[psg.Canvas(key='controls_cv')], psg.Canvas(size=(1000, 500), key='canvas')],
         [psg.Column(column_1, expand_x=True, pad=(0, 0), element_justification='center', background_color='white'),
          psg.Column(column_2, expand_x=True, pad=(0, 0), element_justification='center', background_color='white'),
          psg.Column(column_3, expand_x=True, pad=(0, 0), element_justification='right', background_color='white')]]

# layout = [[psg.Canvas(size=(1000, 500), key='canvas')],
#           [psg.Text("Predictions will be here!", key='predictions'), psg.Text("Once a flare happens, its full prediction history will be here!", key='most_recent_predictions')]]

# launch app
if __name__ == '__main__':

    if not os.path.exists("Artifacts"):
        os.mkdir("Artifacts")

    fig_agg = None

    # Create the Window
    psg.theme("Reddit")
    window = psg.Window('Probably SVEN or MATTIAS, We\'ll See', layout, icon='sven.ico', resizable=True, finalize=True)

    l = App.LiveApp(use_test_flux=False, print_output=False, use_secondary_source=False)
    # window['predictions'].update(format_prediction_string(stop_minute=-3))
    update_current_table()
    l.in_standby_mode = True

    threading.Thread(target=worker_thread1, args=(fig_agg, window,),  daemon=True).start()

    # Event Loop to process "events" and get the "values" of the inputs
    while True:
        event, values = window.read()

        if event == "-THREAD-":
            # time.sleep(1)
            if fig_agg is not None:
                delete_fig_agg(fig_agg)
            # fig = fig_maker(window)
            # fig_agg = draw_figure(window['canvas'].TKCanvas, fig)
            window.Refresh()

        elif event == "ForceStandbyMode":

            l.select_mode("Standby")
            l.exit_flare_mode()
            l.minutes_since_start = -2
            l.standby_mode()

        elif event == "Help":

            show_help()

        # if user closes window or clicks cancel
        if event == psg.WIN_CLOSED or event == 'Cancel':
            # create output folder for flare
            l.save_seven_day_flux_history_json()
            break

    window.close()