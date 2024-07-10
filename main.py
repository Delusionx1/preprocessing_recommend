from PySide6.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QWidget, QPushButton, QTextEdit, QFileDialog, QMessageBox, QVBoxLayout
from PySide6.QtGui import QAction
import mne,sys

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("Analyzing tool")

        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("files")
        about_menu = menu_bar.addMenu("About")

        import_action = QAction("Importing files", self)
        import_action.triggered.connect(self.import_file)
        file_menu.addAction(import_action)

        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about_dialog)
        about_menu.addAction(about_action)

        central_widget = QWidget()
        h_layout = QHBoxLayout(central_widget)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        analyze_button = QPushButton("Analysis of the recommended \n preprocessing method sequence")
        analyze_button.clicked.connect(self.analyze)
        left_layout.addWidget(analyze_button)

        exit_button = QPushButton("Quit")
        exit_button.clicked.connect(self.close)
        left_layout.addWidget(exit_button)

        self.result_text_edit = QTextEdit()
        self.result_text_edit.setReadOnly(True)

        h_layout.addWidget(left_widget)
        h_layout.addWidget(self.result_text_edit)

        self.setCentralWidget(central_widget)

    def import_file(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Importing files", "", "EDF files (*.edf);;FIF files (*.fif);;All files (*.*)")
        if file_path:
            try:
                if file_path.endswith('.edf'):
                    self.raw = mne.io.read_raw_edf(file_path, preload=True)
                elif file_path.endswith('.fif'):
                    self.raw = mne.io.read_raw_fif(file_path, preload=True)
                else:
                    self.result_text_edit.setPlainText("Unsupported file format")
                    return

                channel_names = self.raw.ch_names
                channel_count = len(channel_names)

                result_text = f"Imported file path: {file_path}\n"
                result_text += f"Number of channels: {channel_count}\n"
                result_text += f"Channel Name:\n" + "\n".join(channel_names)

                self.result_text_edit.setPlainText(result_text)
            except Exception as e:
                self.result_text_edit.setPlainText(f"Error importing file: {str(e)}")

    def show_about_dialog(self):
        QMessageBox.about(self, "About:\n", "This is an application that recommends a preprocessing pipeline for analyzing EEG data.\n It is based on an academic paper by Professor Dingguo Zhang’s team \
                          at the University of Bath: ‘Effects of Different Preprocessing Pipelines on Motor Imagery-Based Brain-Computer Interfaces’\n Authors: Xin Gao, \n \tDr. Xiaolong Wu, \n \tDr. Benjamin Metcalfe, \n \tDr. Dingguo Zhang*")

    def analyze(self):
        try:
            if len(self.raw.ch_names) > 32:
                code_text = (
                    "Recommended order of preprocessing methods:\n"
                    "1. Baseline correction\n"
                    "```python\n"
                    "def baseline_correction(epoched_data):\n"
                    "    return epoched_data.apply_baseline(baseline=(None, 0.5))\n"
                    "```\n\n"
                    "2. Bandpass filtering\n"
                    "```python\n"
                    "def bandpass_filtering(epoched_data, Fs=250, notch_freq=50):\n"
                    "    montage = epoched_data.get_montage()\n"
                    "    info = epoched_data.info\n"
                    "    data_notched = epoched_data.get_data()\n"
                    "    data_notched = mne.filter.notch_filter(data_notched, Fs=Fs, freqs=notch_freq)\n"
                    "    epoched_data = mne.EpochsArray(data=data_notched, info=info)\n"
                    "    epoched_data.set_montage(montage, match_case=False)\n"
                    "    return epoched_data.filter(l_freq=1., h_freq=50, method='iir')\n"
                    "```\n\n"
                    "3. Surface Laplace\n"
                    "```python\n"
                    "def surface_laplacian(epoched_data):\n"
                    "    m = 4\n"
                    "    leg_order = 50\n"
                    "    smoothing = 1e-5\n"
                    "    montage = epoched_data.get_montage()\n\n"
                    "    locs = epoched_data._get_channel_positions()\n"
                    "    x = locs[:, 0]\n"
                    "    y = locs[:, 1]\n"
                    "    z = locs[:, 2]\n\n"
                    "    data = epoched_data.get_data()\n"
                    "    data = np.rollaxis(data, 0, 3)\n"
                    "    orig_data_size = np.squeeze(data.shape)\n\n"
                    "    numelectrodes = len(x)\n\n"
                    "    def cart2sph(x, y, z):\n"
                    "        hxy = np.hypot(x, y)\n"
                    "        r = np.hypot(hxy, z)\n"
                    "        el = np.arctan2(z, hxy)\n"
                    "        az = np.arctan2(y, x)\n"
                    "        return az, el, r\n\n"
                    "    junk1, junk2, spherical_radii = cart2sph(x, y, z)\n"
                    "    maxrad = np.max(spherical_radii)\n"
                    "    x = x / maxrad\n"
                    "    y = y / maxrad\n"
                    "    z = z / maxrad\n\n"
                    "    cosdist = np.zeros((numelectrodes, numelectrodes))\n"
                    "    for i in range(numelectrodes):\n"
                    "        for j in range(i + 1, numelectrodes):\n"
                    "            cosdist[i, j] = 1 - (((x[i] - x[j])**2 + (y[i] - y[j])**2 + (z[i] - z[j])**2) / 2)\n\n"
                    "    cosdist = cosdist + cosdist.T + np.identity(numelectrodes)\n\n"
                    "    legpoly = np.zeros((leg_order, numelectrodes, numelectrodes))\n"
                    "    for ni in range(leg_order):\n"
                    "        for i in range(numelectrodes):\n"
                    "            for j in range(i + 1, numelectrodes):\n"
                    "                legpoly[ni, i, j] = special.lpn(ni + 1, cosdist[i, j])[0][ni + 1]\n\n"
                    "    legpoly = legpoly + np.transpose(legpoly, (0, 2, 1))\n\n"
                    "    for i in range(leg_order):\n"
                    "        legpoly[i, :, :] = legpoly[i, :, :] + np.identity(numelectrodes)\n\n"
                    "    twoN1 = np.multiply(2, range(1, leg_order + 1)) + 1\n"
                    "    gdenom = np.power(np.multiply(range(1, leg_order + 1), range(2, leg_order + 2)), m, dtype=float)\n"
                    "    hdenom = np.power(np.multiply(range(1, leg_order + 1), range(2, leg_order + 2)), m - 1, dtype=float)\n\n"
                    "    G = np.zeros((numelectrodes, numelectrodes))\n"
                    "    H = np.zeros((numelectrodes, numelectrodes))\n\n"
                    "    for i in range(numelectrodes):\n"
                    "        for j in range(i, numelectrodes):\n"
                    "            g = 0\n"
                    "            h = 0\n"
                    "            for ni in range(leg_order):\n"
                    "                g = g + (twoN1[ni] * legpoly[ni, i, j]) / gdenom[ni]\n"
                    "                h = h - (twoN1[ni] * legpoly[ni, i, j]) / hdenom[ni]\n"
                    "            G[i, j] = g / (4 * math.pi)\n"
                    "            H[i, j] = -h / (4 * math.pi)\n\n"
                    "    G = G + G.T\n"
                    "    H = H + H.T\n\n"
                    "    G = G - np.identity(numelectrodes) * G[1, 1] / 2\n"
                    "    H = H - np.identity(numelectrodes) * H[1, 1] / 2\n\n"
                    "    if np.any(orig_data_size == 1):\n"
                    "        data = data[:]\n"
                    "    else:\n"
                    "        data = np.reshape(data, (orig_data_size[0], np.prod(orig_data_size[1:3])))\n\n"
                    "    Gs = G + np.identity(numelectrodes) * smoothing\n"
                    "    GsinvS = np.sum(np.linalg.inv(Gs), 0)\n"
                    "    dataGs = np.dot(data.T, np.linalg.inv(Gs))\n"
                    "    C = dataGs - np.dot(np.atleast_2d(np.sum(dataGs, 1) / np.sum(GsinvS)).T, np.atleast_2d(GsinvS))\n\n"
                    "    original = np.reshape(data, orig_data_size)\n"
                    "    surf_lap = np.reshape(np.transpose(np.dot(C, np.transpose(H))), orig_data_size)\n\n"
                    "    events = epoched_data.events\n"
                    "    event_id = epoched_data.event_id\n"
                    "    ch_names = epoched_data.ch_names\n"
                    "    sfreq = epoched_data.info['sfreq']\n"
                    "    tmin = epoched_data.tmin\n"
                    "    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')\n\n"
                    "    original = np.rollaxis(original, 2, 0)\n"
                    "    surf_lap = np.rollaxis(surf_lap, 2, 0)\n\n"
                    "    before = mne.EpochsArray(data=original, info=info, events=events, event_id=event_id, tmin=tmin, on_missing='ignore')\n"
                    "    before.set_montage(montage, match_case=False)\n"
                    "    after = mne.EpochsArray(data=surf_lap, info=info, events=events, event_id=event_id, tmin=tmin, on_missing='ignore')\n"
                    "    after.set_montage(montage, match_case=False)\n\n"
                    "    return after\n"
                    "```\n"
                )
                self.result_text_edit.setPlainText(code_text)
            else:
                code_text = (
                    "Recommended order of preprocessing methods:\n"
                    "1. Baseline correction\n"
                    "```python\n"
                    "def baseline_correction(epoched_data):\n"
                    "    return epoched_data.apply_baseline(baseline=(None, 0.5))\n"
                    "```\n\n"
                    "2. Bandpass filtering\n"
                    "```python\n"
                    "def bandpass_filtering(epoched_data, Fs=250, notch_freq=50):\n"
                    "    montage = epoched_data.get_montage()\n"
                    "    info = epoched_data.info\n"
                    "    data_notched = epoched_data.get_data()\n"
                    "    data_notched = mne.filter.notch_filter(data_notched, Fs=Fs, freqs=notch_freq)\n"
                    "    epoched_data = mne.EpochsArray(data=data_notched, info=info)\n"
                    "    epoched_data.set_montage(montage, match_case=False)\n"
                    "    return epoched_data.filter(l_freq=1., h_freq=50, method='iir')\n"
                    "```\n"
                    " Or you can choose to use raw data")
                self.result_text_edit.setPlainText(code_text)
        except Exception as e:
            self.result_text_edit.setPlainText(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
