from time import sleep
from typing import Callable, Union

import pyperclip
import customtkinter as ctk

from aitunes.user_controls.headless import HeadlessActionPipeline 


gui: 'GUI' = None
_pipeline: Union[None, HeadlessActionPipeline] = None

class GUI(ctk.CTk):

    def __init__(self):
        global gui, _pipeline
        assert gui is None
        
        super().__init__()
        gui = self

        if _pipeline is None:
            _pipeline = HeadlessActionPipeline()
        self.actions = _pipeline

        self.view = None
        self.set_view(MainView(self, self.actions), first=True)
        self.set_base_config()
        
    def fullscreen(self):
        self.after(1, lambda: self.state('zoomed'))

    def set_base_config(self):
        self.title("AiTunes Control Panel")
        self.geometry("1200x800")
        self.fullscreen()

        # Configure grid
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
    
    def set_view(self, view: ctk.CTkFrame, first: bool = False):
        if not first:
            GUIUtils.clear(self)
        view.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.view = view


class MainView(ctk.CTkFrame):

    def __init__(self, gui: ctk.CTkFrame, actions: HeadlessActionPipeline, **kwargs):
        super().__init__(gui, **kwargs)
        self.gui = gui
        self.actions = actions
        self._refresh()

    def _get_shared_dict(self) -> dict:
        return {
            "epochs": self.set_epochs_slider.get(),
            "save_every": self.set_save_period_slider.get(),
            "release_only": self.exp_pick_model_release_cb.get() == 1
        }
    
    def _run_closed(self, fn):
        # global _shared_state
        # _shared_state = self._get_shared_dict()
        GUIUtils.run_closed(fn, self._read_state_to_widgets)

    def _refresh(self):
        GUIUtils.clear(self)
        self._create_main_title()
        self._create_top_level_frames()
        self._create_action_buttons()
        self._create_experiment_selection_pane()
        self._create_experiment_settings_pane()
        self._create_code_output_pane()
        self._read_state_to_widgets()

    def _create_top_level_frames(self):
        self.btn_frame = ctk.CTkFrame(self, fg_color="#101010", height=45)
        self.btn_frame.pack(side=ctk.BOTTOM, padx=10, pady=10, fill=ctk.X)

        self.left_frame = ctk.CTkFrame(self)
        self.left_frame.pack(side=ctk.LEFT, padx=10, fill=ctk.BOTH, expand=True)

        self.right_frame = ctk.CTkFrame(self)
        self.right_frame.pack(side=ctk.RIGHT, padx=10, fill=ctk.BOTH, expand=True)

    def _create_main_title(self):
        # self.grid_rowconfigure(0, weight=1)
        self.main_title = ctk.CTkLabel(self, text="AiTunes Control Panel", font=("Helvetica", 38))
        self.main_title.pack(side=ctk.TOP, anchor="n", pady=5)

        self.disclaimer_label = ctk.CTkLabel(self, text="DISCLAIMER: The intended purpose for this interface is to facilitate the process of switching models.\nRunning training sessions from this interface will inevitably lead to poor performance due to GPU and RAM overhead", font=("Helvetica", 12), text_color="#e63939")
        self.disclaimer_label.pack(side=ctk.TOP, anchor="n", pady=5)
    
    def _create_experiment_selection_pane(self):
        self.exp_frame = self.left_frame
        self.exp_title = ctk.CTkLabel(self.exp_frame, text="Experiment", font=("Helvetica", 26))
        self.exp_title.pack(anchor="n", pady=10)
    
        self.exp_pick = ctk.CTkFrame(self.exp_frame, fg_color="transparent")
        self.exp_pick_label = ctk.CTkLabel(self.exp_pick, text="Experiment Case:")
        self.exp_pick_label.pack(side=ctk.LEFT)
        self.exp_pick_cb = ctk.CTkComboBox(self.exp_pick, state="readonly", values=self.actions.list_scripted_experiments(), command=self._write_widgets_to_state)
        self.exp_pick_cb.set(self.exp_pick_cb._values[0])
        self.exp_pick_cb.pack(side=ctk.RIGHT)
        self.exp_pick.pack(side=ctk.TOP, padx=10, pady=10, fill=ctk.X)

        self.exp_pick_scenario = ctk.CTkFrame(self.exp_frame, fg_color="transparent")
        self.exp_pick_scenario_label = ctk.CTkLabel(self.exp_pick_scenario, text="Scenario:")
        self.exp_pick_scenario_label.pack(side=ctk.LEFT)
        self.exp_pick_scenario_cb = ctk.CTkComboBox(self.exp_pick_scenario, state="disabled", command=self._write_widgets_to_state)
        self.exp_pick_scenario_cb.pack(side=ctk.RIGHT)
        self.exp_pick_scenario.pack(side=ctk.TOP, padx=10, fill=ctk.X)

        self.exp_pick_model = ctk.CTkFrame(self.exp_frame, fg_color="transparent")
        self.exp_pick_model_label = ctk.CTkLabel(self.exp_pick_model, text="Model:")
        self.exp_pick_model_label.pack(side=ctk.LEFT)
        self.exp_pick_model_cb = ctk.CTkComboBox(self.exp_pick_model, state="disabled", command=self._write_widgets_to_state)
        self.exp_pick_model_cb.pack(side=ctk.RIGHT)
        self.exp_pick_model_release_cb = ctk.CTkCheckBox(self.exp_pick_model, text="Latest Release", font=("Helvetica", 12), command=self._write_widgets_to_state)
        self.exp_pick_model_release_cb.select()
        self.exp_pick_model_release_cb.pack(side=ctk.RIGHT, padx=10)
        self.exp_pick_model.pack(side=ctk.TOP, padx=10, pady=10, fill=ctk.X)

        self.exp_description = ctk.CTkFrame(self.exp_frame, fg_color="#202020")
        self.exp_description_head = ctk.CTkLabel(self.exp_description, anchor="w", text="Description", font=("Helvetica", 15, "bold"))
        self.exp_description_head.pack(side=ctk.TOP, padx=10, anchor="w")
        self.exp_description_cnt = ctk.CTkLabel(self.exp_description, anchor="w", justify="left", compound="left", text="Nothing to show", font=("Courier", 14, "bold"))
        self.exp_description_cnt.pack(side=ctk.TOP, pady=5, padx=20, anchor="w")
        self.exp_description.pack(side=ctk.LEFT, padx=10, pady=10, ipady=10, fill=ctk.BOTH, expand=True)

        self.exp_frame.bind("<Configure>", lambda ev: self.exp_description_cnt.configure(wraplength=int(0.75 * ev.width)))
        
    def _create_experiment_settings_pane(self):
        self.set_frame = ctk.CTkFrame(self.right_frame, bg_color="transparent")
        self.set_frame.pack(side=ctk.TOP, padx=10, pady=10, fill=ctk.X)

        self.set_title = ctk.CTkLabel(self.set_frame, text="Training Settings", font=("Helvetica", 26))
        self.set_title.pack(anchor="n", pady=10)
    
        self.set_epochs = ctk.CTkFrame(self.set_frame, fg_color="transparent")
        self.set_epochs_label = ctk.CTkLabel(self.set_epochs, text="Train for:")
        self.set_epochs_label.pack(side=ctk.LEFT)
        self.set_epochs_slider = ctk.CTkSlider(self.set_epochs, from_=0, to=1000, number_of_steps=100, command=self._write_widgets_to_state)
        self.set_epochs_slider.set(0)
        self.set_epochs_slider.pack(side=ctk.RIGHT)
        self.set_epochs_display = ctk.CTkLabel(self.set_epochs, text="0 epoch")
        self.set_epochs_display.pack(side=ctk.RIGHT, padx=10)
        self.set_epochs.pack(side=ctk.TOP, padx=10, pady=10, fill=ctk.X)
        
        self.set_save_period = ctk.CTkFrame(self.set_frame, fg_color="transparent")
        self.set_save_period_label = ctk.CTkLabel(self.set_save_period, text="Save every:")
        self.set_save_period_label.pack(side=ctk.LEFT)
        self.set_save_period_slider = ctk.CTkSlider(self.set_save_period, from_=0, to=100, number_of_steps=20, command=self._write_widgets_to_state)
        self.set_save_period_slider.set(0)
        self.set_save_period_slider.pack(side=ctk.RIGHT)
        self.set_save_period_display = ctk.CTkLabel(self.set_save_period, text="0 epoch")
        self.set_save_period_display.pack(side=ctk.RIGHT, padx=10)
        self.set_save_period.pack(side=ctk.TOP, padx=10, fill=ctk.X)
        
        self.set_plotting_cb = ctk.CTkCheckBox(self.set_frame, text="Plot Epoch Loss", font=("Helvetica", 12), command=self._update_code_displaybox)
        self.set_plotting_cb.deselect()
        self.set_plotting_cb.pack(side=ctk.TOP, anchor="w", pady=10, padx=10)

        self.tooltip_text = ctk.CTkLabel(self.set_frame, text_color="#e63939", text="*No number of epochs selected")
    
    def _create_code_output_pane(self):
        self.code_frame = ctk.CTkFrame(self.right_frame, bg_color="transparent")
        self.code_frame.pack(side=ctk.BOTTOM, padx=10, pady=10, fill=ctk.X)

        self.code_title = ctk.CTkLabel(self.code_frame, text="Code Output", font=("Helvetica", 26))
        self.code_title.pack(anchor="n", pady=10)

        def copy_action():
            cnt = self.code_display_text.get("1.0", ctk.END)
            pyperclip.copy(cnt)
            self.code_copy_btn.configure(text="Copied !", fg_color="#32a852", hover_color="#24783a")
            self.code_copy_btn.after(2000, lambda: self.code_copy_btn.configure(text="Copy code", fg_color="#1F6AA5", hover_color="#0E3D6A"))

        self.code_copy_btn = ctk.CTkButton(self.code_frame, text="Copy Code", height=25, command=copy_action)
        self.code_copy_btn.pack(anchor="n", pady=5)

        self.code_display_text = ctk.CTkTextbox(self.code_frame)
        self.code_display_text.pack(anchor="n", pady=5, padx=5, fill=ctk.BOTH, expand=True)
    
    def _create_action_buttons(self):
        def train_command():
            self._run_closed(lambda: self.actions.train(
                epochs=int(self.set_epochs_slider.get()),
                save_period=int(self.set_save_period_slider.get()),
                plot_progress=self.set_plotting_cb.get() == 1
            ))
        
        def eval_command():
            self._run_closed(lambda: self.actions.evaluate())
        
        def interactive_eval_command():
            self._run_closed(lambda: self.actions.interactive_evaluation())

        self.btn_train = ctk.CTkButton(self.btn_frame, text="Start Training", height=35, command=train_command)
        self.btn_train.pack(side=ctk.RIGHT, padx=10, pady=10)

        self.btn_eval = ctk.CTkButton(self.btn_frame, text="Run Evaluation", height=35, command=eval_command)
        self.btn_eval.pack(side=ctk.RIGHT, pady=10)

        self.btn_interactive_eval = ctk.CTkButton(self.btn_frame, text="Interactive Evaluation", height=35, command=interactive_eval_command)
        self.btn_interactive_eval.pack(side=ctk.RIGHT, padx=10, pady=10)

        self.btn_quit = ctk.CTkButton(self.btn_frame, text="Exit", height=35, command=self.gui.quit)
        self.btn_quit.configure(fg_color='#e63939', hover_color='#8f2727')
        self.btn_quit.pack(side=ctk.LEFT, padx=10, pady=10)

        self.btn_refresh = ctk.CTkButton(self.btn_frame, text="Refresh", height=35, command=self._read_state_to_widgets)
        self.btn_refresh.configure(fg_color='#e68d39', hover_color='#8f4d27')
        self.btn_refresh.pack(side=ctk.LEFT, pady=10)
    
    ### This was created before knowing I could simply .iconify the tkinter window.
    def _read_state_to_widgets(self, *_):
        global _shared_state
        GUIUtils.set_all_states(self, "disabled")
        
        # Clear out scenario and model dropdowns (Updated later in this function if necessary)
        self.exp_pick_scenario_cb.configure(values=[""], state="normal")
        self.exp_pick_scenario_cb.set("---")
        self.exp_pick_scenario_cb.configure(state="disabled")

        self.exp_pick_model_cb.configure(values=[""], state="normal")
        self.exp_pick_model_cb.set("---")
        self.exp_pick_model_cb.configure(state="disabled")

        # Enable Exit and Refresh buttons
        self.btn_quit.configure(state="normal")
        self.btn_refresh.configure(state="normal")
        
        # Reset description
        self.exp_description_cnt.configure(text=self.actions.describe_current_state())

        # Experimentation Case Dropdown
        available_experiments = self.actions.list_scripted_experiments()
        self.exp_pick_cb.configure(state="readonly", values=available_experiments)
        if self.actions.get_selected_experiment() is None:
            self.exp_pick_cb.set("---")
            self._update_code_displaybox()
            return
        
        # Experiment is not None
        self.code_copy_btn.configure(state="normal")
        self.exp_pick_cb.set(self.actions.get_selected_experiment().get_identifier())
        available_scenarios = list(map(lambda x: x.identifier, self.actions.list_scenarios()))
        self.exp_pick_scenario_cb.configure(state="readonly", values=available_scenarios)
        if self.actions.get_selected_scenario() is None:
            self._update_code_displaybox()
            return
        
        # Scenario is not None (Should never be in GUI mode)
        self.exp_pick_scenario_cb.set(self.actions.get_selected_scenario().identifier)
        available_models = self.actions.list_models()
        selected_model = self.actions.get_selected_model()

        # Show the proper selected model
        if len(available_models) > 0:  # At least one model found, fill in the values
            # Update the checkbox and actually WRITE to the state if all conditions match
            self.exp_pick_model_release_cb.configure(state="normal")  # Enable the checkbox
            latest_only = self.exp_pick_model_release_cb.get() == 1
            if latest_only:
                # Checkbox is checked, update the selected model
                self.actions.select_model(available_models[0])
                selected_model = available_models[0]

            # Update the model dropdown
            self.exp_pick_model_cb.configure(state="readonly", values=available_models)
            if selected_model is not None:  # Should never be none at this point
                self.exp_pick_model_cb.set(selected_model)
            
            # Disable it if the checkbox is checked
            if latest_only:
                self.exp_pick_model_cb.configure(state="disabled")
        
        # Enable and set setting controls
        GUIUtils.set_all_states(self.right_frame, "normal")

        epochs_val = int(self.set_epochs_slider.get())
        period_val = int(self.set_save_period_slider.get())
        self.set_epochs_display.configure(text=str(epochs_val) + " epoch" + ("s" if epochs_val > 1 else ""))
        self.set_save_period_display.configure(text=str(period_val) + " epoch" + ("s" if period_val > 1 else ""))

        # Enable buttons
        if epochs_val > 0:
            self.btn_train.configure(state="normal")
            self.tooltip_text.forget()
        else:
            self.tooltip_text.pack(side=ctk.BOTTOM, pady=10, padx=5)

        if selected_model is not None:
            self.btn_interactive_eval.configure(state="normal")
            self.btn_eval.configure(state="normal")

        self._update_code_displaybox()
    
    def _clear_code_displaybox(self):
        self.code_display_text.configure(state="normal")
        self.code_display_text.delete("1.0", ctk.END)
        self.code_display_text.configure(state="disabled")

    def _update_code_displaybox(self, *_):
        self.code_display_text.configure(state="normal")
        self.code_display_text.delete("1.0", ctk.END)
        self.code_display_text.insert("1.0", self.actions.get_current_code(
            int(self.set_epochs_slider.get()),
            int(self.set_save_period_slider.get()),
            self.set_plotting_cb.get() == 1
        ))
        self.code_display_text.configure(state="disabled")

    def _write_widgets_to_state(self, *_):
        selected_experiment = self.exp_pick_cb.get()
        if not self.actions.select_experiment(selected_experiment):
            self.actions.close_experiment()
            self._read_state_to_widgets()
            return
        
        selected_scenario = self.exp_pick_scenario_cb.get()
        if not self.actions.select_scenario(selected_scenario):
            self.actions.close_scenario()
            self._read_state_to_widgets()
            return
        
        all_models = self.actions.list_models()
        selected_model = self.exp_pick_model_cb.get()
        if len(all_models) > 0 and self.exp_pick_model_release_cb.get() == 1:
            selected_model = all_models[0]
        
        if not self.actions.select_model(selected_model):
            self.actions.close_model()
            self._read_state_to_widgets()
            return
        
        self._read_state_to_widgets()


class GUIUtils:

    @staticmethod
    def clear(root):
        for widget in root.winfo_children():
            widget.destroy()
    
    @staticmethod
    def set_all_states(container, state):
        for widget in container.winfo_children():
            if isinstance(widget, (ctk.CTk, ctk.CTkFrame)):
                GUIUtils.set_all_states(widget, state)
                continue
            if getattr(widget, "_state", None) is not None:
                widget.configure(state=state)
    
    @staticmethod
    def run_closed(fn, then: Union[Callable, None]):
        global gui
        gui.withdraw()
        try:
            fn()
        finally:
            gui.deiconify()
            gui.fullscreen()
            then()
