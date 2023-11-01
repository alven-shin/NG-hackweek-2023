import dearpygui.dearpygui as dpg


def main() -> None:
    dpg.create_context()
    dpg.create_viewport(title="Custom Title", width=600, height=600)

    gui()

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()


def save_callback():
    dpg.add_file_dialog(height=300)


def gui():
    with dpg.window(label="Example Window"):
        dpg.add_text("Hello world")
        dpg.add_button(label="Save", callback=save_callback)
        dpg.add_input_text(label="string")
        dpg.add_slider_float(label="float")


if __name__ == "__main__":
    main()
