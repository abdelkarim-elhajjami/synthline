import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, Any, List

from llm_client import LLMClient
from promptline import Promptline
from output import Output
from generator import Generator
from fm import FM, Feature

class GUI:
    """Main GUI class for Synthline application."""
    
    def __init__(self, deepseek_key: str, openai_key: str) -> None:
        # Core components
        self._features: Dict[str, Feature] = FM().features
        self._llm: LLMClient = LLMClient(deepseek_key, openai_key)
        self._promptline: Promptline = Promptline()
        self._generator: Generator = Generator(self._llm, self._promptline)
        self._output: Output = Output()
        
        # GUI components
        self._root: tk.Tk = tk.Tk()
        self._feature_widgets: Dict[str, Any] = {}
        self._dynamic_frames: Dict[str, tuple] = {}
        
        self._root.title("Synthline")
        self._root.geometry("700x800")
        
        # Create canvas with scrollbar
        self._canvas = tk.Canvas(self._root)
        self._scrollbar = ttk.Scrollbar(self._root, orient="vertical", 
                                      command=self._canvas.yview)
        self._scrollable_frame = ttk.Frame(self._canvas)
        
        # Configure canvas
        self._scrollable_frame.bind(
            "<Configure>",
            lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        )
        self._canvas.create_window((0, 0), window=self._scrollable_frame, anchor="nw")
        self._canvas.configure(yscrollcommand=self._scrollbar.set)
        
        # Pack scrollbar components
        self._scrollbar.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)
        
        self._create_widgets()
    
    def _create_widgets(self) -> None:
        main_frame: ttk.Frame = ttk.Frame(self._scrollable_frame, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self._canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # Title
        row = 0
        heading = ttk.Label(main_frame, text="Synthline", font=('Helvetica', 16, 'bold'))
        heading.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=10)
        row += 1

        # Create main sections
        sections = [
            ('Generator', 'generator'),
            ('Artifact', 'artifact'),
            ('ML Task', 'ml_task'),
            ('Output', 'output')
        ]

        for section_title, feature_name in sections:
            row = self._create_section(main_frame, section_title, self._features[feature_name], row)

        # Status and Generate Button
        row = self._create_action_section(main_frame, row)

    def _create_section(self, parent: ttk.Frame, title: str, feature: Feature, row: int, padding: int = 40) -> int:
        """
        Create a section in the GUI for a given feature.
        """
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # Configure columns for consistent alignment
        frame.grid_columnconfigure(0, minsize=200)
        frame.grid_columnconfigure(1, weight=1)
        
        if feature.feature_type == 'select':
            # Create header with combobox on the same line
            heading = ttk.Label(frame, text=title, font=('Helvetica', 14, 'bold'))
            heading.grid(row=0, column=0, sticky=tk.W, pady=10)
            
            var = tk.StringVar()
            widget = ttk.Combobox(frame, textvariable=var, values=feature.options, state="readonly")
            widget.grid(row=0, column=1, sticky=tk.E, pady=10, padx=(0, 20))  # Add right padding
            
            # Create frame for subfeatures (initially empty)
            subfeatures_frame = ttk.Frame(parent)
            subfeatures_frame.grid(row=row+1, column=0, columnspan=2, sticky=(tk.W, tk.E))
            subfeatures_frame.grid_remove()
            
            # Configure subfeature frame columns to match parent
            subfeatures_frame.grid_columnconfigure(0, minsize=200)
            subfeatures_frame.grid_columnconfigure(1, weight=1)
            
            self._feature_widgets[feature.name] = var
            self._dynamic_frames[feature.name] = (subfeatures_frame, feature.subfeatures)
            
            widget.bind('<<ComboboxSelected>>', 
                       lambda e, f=feature.name: self._on_parent_selected(f))
            row += 2
        else:
            heading = ttk.Label(frame, text=title, font=('Helvetica', 14, 'bold'))
            heading.grid(row=0, column=0, sticky=tk.W, pady=10)
            row += 1
            row = self._create_subfeatures(parent, feature.subfeatures, row, padding)

        return row

    def _create_subfeatures(self, parent: ttk.Frame, subfeatures: Dict[str, Feature], 
                           row: int, padding: int = 40) -> int:
        if not subfeatures:
            return row
        
        subfeature_frame = ttk.Frame(parent)
        subfeature_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # Configure columns for consistent alignment
        subfeature_frame.grid_columnconfigure(0, minsize=200)
        subfeature_frame.grid_columnconfigure(1, weight=1)
        
        inner_row = 0
        for name, subfeature in subfeatures.items():
            if subfeature.feature_type == 'group':
                inner_row = self._create_subfeatures(subfeature_frame, subfeature.subfeatures, inner_row, padding)
            else:
                container = ttk.Frame(subfeature_frame)
                container.grid(row=inner_row, column=0, columnspan=2, sticky=(tk.W, tk.E))
                container.grid_columnconfigure(0, minsize=200)
                container.grid_columnconfigure(1, weight=1)
                
                # Add hint for comma-separated inputs
                if name in ['domain', 'language']:
                    label_text = f"{subfeature.name} (comma-separated)"
                else:
                    label_text = subfeature.name
                
                label = ttk.Label(container, text=label_text)
                label.grid(row=0, column=0, sticky=tk.W, pady=5, padx=(padding, 0))
                
                if subfeature.multiple:
                    widget = MultiSelectCombobox(container, subfeature.options)
                elif subfeature.feature_type == 'select':
                    var = tk.StringVar()
                    widget = ttk.Combobox(container, textvariable=var, values=subfeature.options, state="readonly")
                    self._feature_widgets[name] = var
                else:
                    var = tk.StringVar()
                    widget = ttk.Entry(container, textvariable=var)
                    self._feature_widgets[name] = var
                
                widget.grid(row=0, column=1, sticky=tk.E, pady=5, padx=(0, 20))
                if isinstance(widget, MultiSelectCombobox):
                    self._feature_widgets[name] = widget
                inner_row += 1
        
        return row + 1

    def _on_parent_selected(self, feature_name: str) -> None:
        frame, subfeatures = self._dynamic_frames[feature_name]
        selected = self._feature_widgets[feature_name].get().lower()
        
        for widget in frame.winfo_children():
            widget.destroy()
            
        if selected in subfeatures:
            frame.grid()
            selected_feature = subfeatures[selected]
            if selected_feature.subfeatures:
                self._create_subfeatures(frame, selected_feature.subfeatures, 0)
        else:
            frame.grid_remove()

    def _create_action_section(self, parent: ttk.Frame, row: int) -> int:
        separator: ttk.Separator = ttk.Separator(parent, orient='horizontal')
        separator.grid(row=row, column=0, columnspan=2, sticky="ew", pady=10)
        row += 1

        # Add progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(parent, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=20, pady=5)
        row += 1

        self.status_label = ttk.Label(parent, text="", font=('Helvetica', 10))
        self.status_label.grid(row=row, column=0, columnspan=2, pady=5)
        row += 1

        self.generate_btn = ttk.Button(
            parent, 
            text="Generate",
            command=self.generate,
            style='Generate.TButton'
        )
        self.generate_btn.grid(row=row, column=0, columnspan=2, pady=20)

        style = ttk.Style()
        style.configure('Generate.TButton', font=('Helvetica', 11, 'bold'))
        
        return row

    def _on_mousewheel(self, event: tk.Event) -> None:
        self._canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
    
    def _get_feature_values(self) -> Dict[str, Any]:
        """Collect all feature values from the GUI widgets."""
        values = {}
        
        def collect_values(features):
            for name, feature in features.items():
                if feature.subfeatures:
                    collect_values(feature.subfeatures)
                
                # Get the widget associated with this feature name
                widget = self._feature_widgets.get(name)
                if widget:
                    if isinstance(widget, MultiSelectCombobox):
                        values[name] = widget.get_selections()
                    else:
                        value = widget.get().strip()
                        # Handle domain and language as comma-separated lists
                        if name in ['domain', 'language']:
                            values[name] = [item.strip() for item in value.split(',')]
                        else:
                            values[name] = value
                            
        collect_values(self._features)
        return values
    
    def _validate_inputs(self, values: Dict[str, Any]) -> None:
        """Validate all user inputs before generation"""
        # Validate subset size
        try:
            subset_size = int(values['subset_size'])
            if subset_size <= 0:
                raise ValueError("Subset size must be a positive integer")
        except ValueError:
            raise ValueError("Subset size must be a valid integer")

        # Validate temperature
        try:
            temp = float(values['temperature'])
            if not 0 <= temp <= 2:
                raise ValueError("Temperature must be between 0 and 2")
        except ValueError:
            raise ValueError("Temperature must be a valid number")

        # Validate top_p
        try:
            top_p = float(values['top_p'])
            if not 0 <= top_p <= 1:
                raise ValueError("Top P must be between 0 and 1")
        except ValueError:
            raise ValueError("Top P must be a valid number")

        # Validate required text fields are not empty
        required_fields = ['domain', 'language', 'label', 'label_description']
        for field in required_fields:
            if not values[field]:
                raise ValueError(f"{self._features[field].name} cannot be empty")

    def generate(self) -> None:
        try:
            values = self._get_feature_values()
            
            # Validate inputs before proceeding
            self._validate_inputs(values)
            
            # Update UI
            self.generate_btn.configure(state='disabled')
            self.status_label.config(text="Generating...")
            self.progress_var.set(0)  # Reset progress bar
            self._root.update()
            
            def update_progress(value: float) -> None:
                self.progress_var.set(value)
                self._root.update()
            
            # Generate samples with progress tracking
            samples = self._generator.generate_samples(
                subset_size=int(values['subset_size']),
                feature_values=values,
                progress_callback=update_progress
            )
            
            # Save and reset UI
            output_path = self._output.save(samples, values['output_format'], values)
            self.generate_btn.configure(state='normal')
            self.status_label.config(text="")
            
            messagebox.showinfo("Success", f"Generated {len(samples)} requirements\nSaved to: {output_path}")
            
        except Exception as e:
            self.generate_btn.configure(state='normal')
            self.status_label.config(text="")
            messagebox.showerror("Error", str(e))
    
    def run(self) -> None:
        """Public interface to start the GUI."""
        self._root.mainloop()

class MultiSelectCombobox(ttk.Frame):
    def __init__(self, parent: ttk.Widget, options: List[str], **kwargs: Any) -> None:
        super().__init__(parent)
        self.options: List[str] = options
        self.selections: List[str] = []
        
        # Create combobox
        self.combo_var: tk.StringVar = tk.StringVar()
        self.combo: ttk.Combobox = ttk.Combobox(
            self, textvariable=self.combo_var, state="readonly", **kwargs
        )
        self.combo['values'] = self.options
        self.combo.grid(row=0, column=0, sticky="ew")
        
        # Bind selection event
        self.combo.bind('<<ComboboxSelected>>', self._on_select)
        
        # Create selection display
        self.selection_var: tk.StringVar = tk.StringVar()
        self.selection_label: ttk.Label = ttk.Label(
            self, textvariable=self.selection_var, wraplength=200
        )
        self.selection_label.grid(row=1, column=0, sticky="ew")
        
        # Configure grid
        self.columnconfigure(0, weight=1)
        
        self._update_selection_display()
    
    def _on_select(self, event: tk.Event) -> None:
        selected: str = self.combo_var.get()
        if selected in self.selections:
            self.selections.remove(selected)  # Deselect if already selected
        elif selected:
            self.selections.append(selected)  # Select if not already selected
        self._update_selection_display()
        self.combo_var.set('')  # Reset combobox
    
    def _update_selection_display(self) -> None:
        if self.selections:
            self.selection_var.set(", ".join(self.selections))
        else:
            self.selection_var.set("None selected")
    
    def get_selections(self) -> List[str]:
        return self.selections if self.selections else [self.options[0]]