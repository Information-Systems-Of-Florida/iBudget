import sys
import pickle
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QTableView,
    QListWidget, QWidget, QSplitter, QVBoxLayout, QMessageBox, QAction
)
from PyQt5.QtCore import QAbstractTableModel, Qt, QSize

class PandasModel(QAbstractTableModel):
    def __init__(self, df=pd.DataFrame(), parent=None):
        super().__init__(parent)
        self._df = df

    def rowCount(self, parent=None):
        return self._df.shape[0]

    def columnCount(self, parent=None):
        return self._df.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            value = self._df.iloc[index.row(), index.column()]
            return str(value)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._df.columns[section])
            if orientation == Qt.Vertical:
                return str(self._df.index[section])
        return None

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pickle Dictionary Explorer")
        self.setMinimumSize(QSize(900, 600))

        self.splitter = QSplitter(self)
        self.key_list = QListWidget()
        self.table = QTableView()

        self.splitter.addWidget(self.key_list)
        self.splitter.addWidget(self.table)
        self.splitter.setStretchFactor(1, 1)
        self.key_list.setMaximumWidth(250)

        # Central widget and layout
        widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.splitter)
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # Open action
        open_action = QAction("&Open Pickle...", self)
        open_action.triggered.connect(self.load_pickle)
        self.menuBar().addAction(open_action)

        # State
        self.data = None
        self.key_list.itemSelectionChanged.connect(self.display_selected_key)

    def load_pickle(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Pickle File", "", "Pickle Files (*.pkl *.pickle);;All Files (*)")
        if not file_path:
            return
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not open pickle file:\n{e}")
            return

        if not isinstance(data, dict):
            QMessageBox.critical(self, "Error", "This app only supports pickle files containing a dictionary.")
            return

        self.data = data
        self.key_list.clear()
        self.key_list.addItems([str(k) for k in self.data.keys()])
        self.table.setModel(None)

    def display_selected_key(self):
        if self.data is None:
            return
        selected_items = self.key_list.selectedItems()
        if not selected_items:
            return
        key = selected_items[0].text()
        values = self.data[key]
        try:
            # List of dicts ? DataFrame
            if isinstance(values, list) and values and isinstance(values[0], dict):
                value_df = pd.DataFrame(values)
            # Dict ? single-row DataFrame
            elif isinstance(values, dict):
                value_df = pd.DataFrame([values])
            # List/tuple ? single-column DataFrame
            elif isinstance(values, (list, tuple)):
                value_df = pd.DataFrame({'value': values})
            # Scalar ? single-row DataFrame
            else:
                value_df = pd.DataFrame({'value': [values]})
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not display value for key '{key}':\n{e}")
            self.table.setModel(None)
            return

        model = PandasModel(value_df)
        self.table.setModel(model)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
