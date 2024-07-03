import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import numpy as np

class PszApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Psz App")
        self.root.configure(background='orange')

        # Load the models
        self.load_models()

        # Create the GUI components
        self.create_widgets()

    def load_models(self):
        try:
            # Load your models here
            self.customModel = joblib.load('custom_linear_regression_model.pkl')            
            self.builtInModel = joblib.load('linear_regression_model.pkl')
            self.scaler = joblib.load('scaler.pkl')
            self.publishers_data = joblib.load('publishers_data.pkl')
            self.ovr_model = joblib.load('one_vs_rest_model.pkl')
            self.multinomial_model = joblib.load('multinomial_model.pkl')

            print("Models loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            self.customModel = None
            self.builtInModel = None
            self.scaler = None
            self.ovr_model = None
            self.multinomial_model = None

    def create_widgets(self):
        # Godina izdavanja
        self.godina_label = tk.Label(self.root, text="Godina izdanja:", background='orange')
        self.godina_label.grid(row=0, column=0, padx=10, pady=(20, 10))
        self.godina = tk.Entry(self.root, validate="key", validatecommand=(self.root.register(self.validate_integer), '%P'))
        self.godina.grid(row=0, column=1, padx=10, pady=10)

        # Broj strana
        self.broj_strana_label = tk.Label(self.root, text="Broj strana:", background='orange')
        self.broj_strana_label.grid(row=1, column=0, padx=10, pady=10)
        self.broj_strana = tk.Entry(self.root, validate="key", validatecommand=(self.root.register(self.validate_integer), '%P'))
        self.broj_strana.grid(row=1, column=1, padx=10, pady=10)

        # Format
        self.format_label = tk.Label(self.root, text="Format:", background='orange')
        self.format_label.grid(row=2, column=0, padx=10, pady=(20, 10))
        self.format_frame = tk.Frame(self.root)
        self.format_frame.grid(row=2, column=1, padx=10, pady=10, columnspan=1)        
        
        # Width
        self.format_width = tk.Entry(self.format_frame, width=6, validate="key", validatecommand=(self.root.register(self.validate_integer), '%P'))
        self.format_width.pack(side=tk.LEFT)

        # 'x'
        self.x_label = tk.Label(self.format_frame, text="x", background='orange')
        self.x_label.pack(side=tk.LEFT)

        # Height
        self.format_height = tk.Entry(self.format_frame,  width=6, validate="key", validatecommand=(self.root.register(self.validate_integer), '%P'))
        self.format_height.pack(side=tk.LEFT)

        # 'cm'
        self.cm_label = tk.Label(self.format_frame, text="cm", background='orange')
        self.cm_label.pack(side=tk.LEFT)
 
        # Srednja cena po izdavacu
        self.izdavac_label = tk.Label(self.root, text="Izdavac:", background='orange')
        self.izdavac_label.grid(row=3, column=0, padx=10, pady=10)
        self.izdavac = tk.Entry(self.root)
        self.izdavac.grid(row=3, column=1, padx=10, pady=10)
        
        # Option menu
        self.tip_poveza_label = tk.Label(self.root, text="Tip poveza:", background='orange')
        self.tip_poveza_label.grid(row=4, column=0, padx=10, pady=10)
        self.option_var = tk.StringVar(self.root)
        self.option_var.set("Tvrd")  # Set default value
        self.tip_poveza = ttk.OptionMenu(self.root, self.option_var, "Tvrd", "Tvrd", "Broš")
        self.tip_poveza.grid(row=4, column=1, padx=10, pady=10)


        self.model_var = tk.StringVar(self.root)
        self.model_var.set("OVR")  # Set default value

        self.radio_ovr = tk.Radiobutton(self.root, text="One-vs-Rest", variable=self.model_var, value="OVR", background='orange')
        self.radio_ovr.grid(row=5, column=0, padx=10, pady=10)

        self.radio_multinomial = tk.Radiobutton(self.root, text="Multinomial", variable=self.model_var, value="Multinomial", background='orange')
        self.radio_multinomial.grid(row=5, column=1, padx=10, pady=10)

        # Submit button
        self.submit_button = tk.Button(self.root, text="Submit", command=self.on_submit)
        self.submit_button.grid(row=6, column=0, columnspan=2, pady=10)

        # Prediction Labels
        self.custom_label = tk.Label(self.root, text="Predikcija Custom Modela: 0")
        self.custom_label.grid(row=7, column=0, columnspan=2, padx=10, pady=(40, 10))

        self.lin_reg_label = tk.Label(self.root, text="Predikcija Modela Linearne Regresije: 0")
        self.lin_reg_label.grid(row=8, column=0, columnspan=2, padx=10, pady=(10, 10))
        
        self.log_reg_label = tk.Label(self.root, text="Predikcija Modela Logisticke Regresije: 0")
        self.log_reg_label.grid(row=9, column=0, columnspan=2, padx=10, pady=(10, 20))        

    def validate_integer(self, value_if_allowed):
        if value_if_allowed.isdigit() or value_if_allowed == "":
            return True
        else:
            return False

    def on_submit(self):
        # Obtain input data
        godina_izdanja = int(self.godina.get())
        broj_strana_knjige = int(self.broj_strana.get())
        tvrd_povez = 1.0 if self.option_var.get() == "Tvrd" else 0.0
        format_area = float(self.format_width.get()) * float(self.format_height.get())
        izdavac = self.izdavac.get().lower()

        # Determine which model to use
        if self.model_var.get() == "OVR":
            log_reg_model = self.ovr_model
        else:
            log_reg_model = self.multinomial_model

        # Fetch mean price for the publisher
        if izdavac in self.publishers_data['izdavac'].values:
            srednja_cena_po_izdavacu = self.publishers_data.loc[self.publishers_data['izdavac'] == izdavac, 'srednja_cena_po_izdavacu'].values[0]
            print(f"The mean price for {izdavac} is: {srednja_cena_po_izdavacu}")
        else:
            srednja_cena_po_izdavacu = self.publishers_data['srednja_cena_po_izdavacu'].mean()
            print(f"{izdavac} not found. Mean price across all publishers is: {srednja_cena_po_izdavacu}")

        # Prepare input data
        input_data = np.array([[godina_izdanja, broj_strana_knjige, format_area, srednja_cena_po_izdavacu, tvrd_povez, 1.0-tvrd_povez]])

        # Scale input data if scaler is loaded
        if self.scaler:
            input_data = self.scaler.transform(input_data)
        else:
            messagebox.showerror("Error", "Scaler not loaded")

        # Make predictions
        custom_predict = self.customModel.predict(input_data)
        lin_reg_predict = self.builtInModel.predict(input_data)
        log_reg_model_predict = log_reg_model.predict(input_data)
        
        if log_reg_model_predict[0] == 0:
            prediction_label = "Veoma jeftina knjiga"
        elif log_reg_model_predict[0] == 1:
            prediction_label = "Jeftina knjiga"
        elif log_reg_model_predict[0] == 2:
            prediction_label = "Skupa knjiga"
        elif log_reg_model_predict[0] == 3:
            prediction_label = "Veoma skupa knjiga"
        else:
            prediction_label = "Unknown"

        # Update labels with predictions
        self.custom_label.config(text=f"Predikcija Custom Modela: {custom_predict[0][0]}")
        self.lin_reg_label.config(text=f"Predikcija Modela Linearne Regresije: {lin_reg_predict[0][0]}")
        self.log_reg_label.config(text=f"Predikcija Modela Logisticke Regresije: {prediction_label}")
        
        print(f"Model selected: {self.model_var.get()}")
        print(f"Model Prediction: {log_reg_model_predict}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PszApp(root)
    root.mainloop()
