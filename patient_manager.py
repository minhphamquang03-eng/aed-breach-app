import pandas as pd
import logging
import matplotlib.pyplot as plt
from datetime import datetime

# ================= LOGGING SETUP =================
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    filename="audit_log.txt",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)


class PatientDataManager:

    MAX_UNDO = 20

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.df_original = self.df.copy()
        self.undo_stack = []

    # ---------- LOG ----------
    def log_action(self, action):
        print(f"Action Logged: {action}")
        logging.info(action)

    # ---------- RETRIEVE ----------
    def get_patient_by_id(self):
        pid = input("Enter patient ID: ")
        result = self.df[self.df["ID"].astype(str) == pid]

        if result.empty:
            print("❌ Patient not found.")
            logging.info(f"Attempted retrieval of non-existent patient ID {pid}")
        else:
            print("\nPatient record:")
            print(result)
            logging.info(f"Retrieved patient ID {pid}")

    # ---------- ADVANCED FILTER (KEEP ONLY FILTER1) ----------
    def advanced_filter1(self):
        filtered = self.df.copy()

        while True:
            print("\nAvailable columns:")
            print(list(filtered.columns))

            col = input("Column to filter (or 'done'): ")
            if col.lower() == "done":
                break

            if col not in filtered.columns:
                print("❌ Invalid column.")
                continue

            if pd.api.types.is_numeric_dtype(filtered[col]):
                try:
                    min_val = float(input("Min value: "))
                    max_val = float(input("Max value: "))
                    filtered = filtered[
                        (filtered[col] >= min_val) & (filtered[col] <= max_val)
                    ]
                except ValueError:
                    print("❌ Invalid number.")
            else:
                print("Unique values:", filtered[col].unique())
                val = input("Enter exact value: ")
                filtered = filtered[filtered[col].astype(str) == val]

            print(f"Remaining rows: {len(filtered)}")

        return filtered

    # ---------- HANDLE FILTERED ----------
    def handle_filtered_dataset(self, df_filtered):
        while True:
            print("""
[1] Set filtered dataset as working dataset
[2] Export filtered dataset to CSV
[3] Export filtered dataset to Excel
[4] Discard and return
""")
            choice = input("Choose option: ")

            if choice == "1":
                self.df = df_filtered.copy()
                logging.info("Set filtered dataset as working dataset")
                print("✅ Working dataset updated.")
                return

            elif choice == "2":
                fname = input("CSV filename: ")
                df_filtered.to_csv(fname, index=False)
                logging.info(f"Exported filtered dataset to {fname}")
                print("✅ CSV exported.")

            elif choice == "3":
                fname = input("Excel filename: ")
                df_filtered.to_excel(fname, index=False)
                logging.info(f"Exported filtered dataset to {fname}")
                print("✅ Excel exported.")

            elif choice == "4":
                print("❌ Changes discarded.")
                return

            else:
                print("❌ Invalid option.")

    # ---------- RESET ----------
    def reset_to_original(self):
        confirm = input("Reset dataset to original? (Y/N): ")
        if confirm.upper() == "Y":
            self.df = self.df_original.copy()
            logging.info("Dataset reset to original")
            print("✅ Dataset reset.")

    # ---------- SAVE STATE ----------
    def save_state(self, action_desc):
        self.undo_stack.append({
            "df": self.df.copy(deep=True),
            "desc": action_desc,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        if len(self.undo_stack) > self.MAX_UNDO:
            self.undo_stack.pop(0)

    # ---------- MODIFY ----------
    def modify_patient(self):
        pid = input("Enter patient ID to modify: ")
        idx = self.df[self.df["ID"].astype(str) == pid].index

        if len(idx) == 0:
            print("❌ Patient not found.")
            return

        print("Available columns:")
        print(list(self.df.columns))

        col = input("Enter column to modify: ")
        if col not in self.df.columns:
            print("❌ Invalid column.")
            return

        old_value = self.df.loc[idx[0], col]
        raw_value = input("Enter new value: ")
        
        col_dtype = self.df[col].dtype
        
        try:
            if pd.api.types.is_integer_dtype(col_dtype):
                new_value = int(raw_value)
                
            elif pd.api.types.is_float_dtype(col_dtype):
                new_value = float(raw_value)
                
            else:
                new_value = raw_value
                
        except ValueError:
            print("❌ Invalid value type for this column.")
            return

        confirm = input(f"Confirm change {col}: {old_value} → {new_value}? (Y/N): ")
        if confirm.upper() == "Y":
            self.save_state(f"Modify patient {pid}: {col}")
            self.df.loc[idx[0], col] = new_value
            logging.info(f"Modified patient ID {pid}: {col} from {old_value} to {new_value}")
            print("✅ Record updated.")
        else:
            print("❌ Modification cancelled.")

    # ---------- DELETE ----------
    def delete_patient(self):
        pid = input("Enter patient ID to delete: ")
        idx = self.df[self.df["ID"].astype(str) == pid].index

        if len(idx) == 0:
            print("❌ Patient not found.")
            return

        confirm = input(f"Are you sure you want to delete patient {pid}? (Y/N): ")
        if confirm.upper() == "Y":
            self.save_state(f"Delete patient {pid}")
            self.df.drop(idx, inplace=True)
            logging.info(f"Deleted patient ID {pid}")
            print("✅ Patient deleted.")
        else:
            print("❌ Deletion cancelled.")

    # ---------- LOG VIEW ----------
    def view_logs(self):
        try:
            with open("audit_log.txt", "r") as f:
                print("\n===== SYSTEM LOG =====")
                print(f.read())
        except FileNotFoundError:
            print("No log file found.")

    # ---------- EXPORT ----------
    def export_data(self):
        print("\nExport options:")
        print("1. CSV")
        print("2. Excel (.xlsx)")
        choice = input("Choose format (1/2): ")
        filename = input("Enter file name (without extension): ")

        if choice == "1":
            self.df.to_csv(f"{filename}.csv", index=False)
            logging.info(f"Exported data to {filename}.csv")
        elif choice == "2":
            self.df.to_excel(f"{filename}.xlsx", index=False)
            logging.info(f"Exported data to {filename}.xlsx")
        else:
            print("❌ Invalid choice.")
            return

        print("✅ Data exported successfully.")

    # ---------- UNDO ----------
    def undo(self):
        if not self.undo_stack:
            print("❌ Nothing to undo.")
            return

        state = self.undo_stack.pop()
        self.df = state["df"]
        logging.info(f"UNDO: {state['desc']}")
        print(f"↩ Undo successful: {state['desc']}")

    def undo_by_choice(self):
        if not self.undo_stack:
            print("❌ Nothing to undo.")
            return

        for i, item in enumerate(self.undo_stack):
            print(f"{i+1}. [{item['time']}] {item['desc']}")

        try:
            choice = int(input("Choose action number to undo: "))
        except ValueError:
            print("❌ Invalid input.")
            return

        if choice < 1 or choice > len(self.undo_stack):
            print("❌ Invalid choice.")
            return

        selected = self.undo_stack[choice - 1]
        confirm = input(f"Undo ALL changes after '{selected['desc']}'? (Y/N): ")

        if confirm.upper() != "Y":
            print("❌ Cancelled.")
            return

        self.df = selected["df"]
        del self.undo_stack[choice - 1:]
        logging.info(f"UNDO TO STATE: {selected['desc']}")
        print("✅ Undo successful.")

    # ---------- MENU ----------
    def menu(self):
        print("\n========== Patient Data Management ==========")
        print("1. Retrieve patient by ID")
        print("2. Modify patient record")
        print("3. Delete patient record")
        print("4. View log file")
        print("5. Export data")
        print("6. Undo last action")
        print("7. Undo by choice")
        print("8. Advance filtering")
        print("9. Reset dataset")
        print("0. Exit")

    # ---------- RUN ----------
    def run(self):
        while True:
            self.menu()
            choice = input("Select option: ")

            if choice == "1":
                self.get_patient_by_id()
            elif choice == "2":
                self.modify_patient()
            elif choice == "3":
                self.delete_patient()
            elif choice == "4":
                self.view_logs()
            elif choice == "5":
                self.export_data()
            elif choice == "6":
                self.undo()
            elif choice == "7":
                self.undo_by_choice()
            elif choice == "8":
                df_filtered = self.advanced_filter1()
                print(df_filtered.head())
                self.handle_filtered_dataset(df_filtered)
            elif choice == "9":
                self.reset_to_original()
            elif choice == "0":
                logging.info("Program exited by user.")
                print("Exited program.")
                break
            else:
                print("❌ Invalid option.")
