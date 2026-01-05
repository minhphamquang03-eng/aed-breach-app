# patient_manager_gui.py
from patient_manager import PatientDataManager

class PatientDataManagerGUI(PatientDataManager):

    def get_patient_by_id_gui(self, pid):
        return self.df[self.df["ID"].astype(str) == str(pid)]

    def modify_patient_gui(self, pid, col, new_value):
        idx = self.df[self.df["ID"].astype(str) == str(pid)].index
        if len(idx) == 0:
            return False, "Patient not found"

        self.save_state(f"Modify patient {pid}: {col}")
        self.df.loc[idx[0], col] = new_value
        return True, "Updated"

    def delete_patient_gui(self, pid):
        idx = self.df[self.df["ID"].astype(str) == str(pid)].index
        if len(idx) == 0:
            return False
        self.save_state(f"Delete patient {pid}")
        self.df.drop(idx, inplace=True)
        return True