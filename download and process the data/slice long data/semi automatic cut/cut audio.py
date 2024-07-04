import sys
import json
import os
import numpy as np
import librosa
import scipy.signal
import soundfile as sf
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTextEdit, QPushButton, 
                             QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, 
                             QMessageBox, QSlider, QLabel, QProgressBar, QComboBox,
                             QLineEdit, QPlainTextEdit)
from PyQt6.QtCore import QUrl, Qt, QTimer, QThread, pyqtSignal, QObject
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from vad import EnergyVAD
import re
import pytube  # Add pytube for YouTube downloads

parsha_names = ["Bereshit", "Noach", "LechLecha", "Vayera", "ChayeiSara", "Toldot", "Vayetzei", "Vayishlach", "Vayeshev", "Miketz", "Vayigash", "Vayechi", "Shemot", "Vaera", "Bo", "Beshalach", "Yitro", "Mishpatim", "Terumah", "Tetzaveh", "KiTisa", "Vayakhel", "Pekudei", "Vayikra", "Tzav", "Shmini", "Tazria", "Metzora", "Acharei Mot", "Kedoshim", "Emor", "Behar", "Bechukotai", "Bamidbar", "Nasso", "Behaalotcha", "Shlach", "Korach", "Chukat", "Balak", "Pinchas", "Matot", "Masei", "Devarim", "Vaethanan", "Eikev", "Reeh", "Shoftim", "KiTeitzei", "KiTavo", "Nitzavim", "Vayeilech", "Haazinu", "VezotHaberakhah"]

class AudioProcessingWorker(QObject):
    progress_updated = pyqtSignal(int)
    processing_finished = pyqtSignal(list, int)

    def __init__(self, audio_file, sr, max_segment_length, output_dir):
        super().__init__()
        self.audio_file = audio_file
        self.sr = sr
        self.max_segment_length = max_segment_length * 1000  # Convert to milliseconds
        self.output_dir = output_dir

    def process_audio(self):
        # Initialize VAD
        FRAME_LENGTH = 20  # in milliseconds
        vad = EnergyVAD(
            sample_rate=self.sr,
            frame_length=FRAME_LENGTH,
            frame_shift=FRAME_LENGTH,
            energy_threshold=0.002,
            pre_emphasis=0.95,
        )
        
        audio, self.sr = librosa.load(self.audio_file, sr=self.sr)
        voice_activity = vad(audio)
        
        # Apply median filter to smooth the voice activity detection
        voice_activity_max_pool = scipy.signal.medfilt(voice_activity, kernel_size=15)
        
        # Split audio into < 30 seconds segments
        segments = []
        segment_files = []
        start = 0  # start of the segment (in frames)
        total_frames = len(voice_activity_max_pool) 

        base_filename = os.path.splitext(os.path.basename(self.audio_file))[0]

        while start < total_frames:
            # Find silence after less than 30 seconds of speech
            for end in range(min(start + self.max_segment_length // FRAME_LENGTH - 1, total_frames), start, -1):
                if end >= len(voice_activity_max_pool):
                    end = len(voice_activity_max_pool) - 1
                if not voice_activity_max_pool[end]:
                    break
            
            segment = audio[start*FRAME_LENGTH*self.sr//1000:end*FRAME_LENGTH*self.sr//1000]
            segments.append(segment)
            
            # Save the segment
            segment_filename = f"{base_filename}_{len(segments):03d}.wav"
            segment_path = os.path.join(self.output_dir, segment_filename)
            sf.write(segment_path, segment, self.sr)
            segment_files.append(segment_filename)
            
            start = end + 1
            
            # Update progress
            progress = int(end / total_frames * 100)
            self.progress_updated.emit(progress)

        self.processing_finished.emit(segment_files, len(segments))
        
    

class AudioSegmentationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Improved Audio Segmentation and Text Synchronization")
        self.setGeometry(100, 100, 800, 600)

        self.audio_file = None
        self.segment_files = []
        self.current_segment = 0
        self.output_dir = ""
        self.project_data = {"text": [], "audio": []}
        self.processing_thread = None
        self.is_playing = False
        self.segment_duration = 0
        self.sr = 16000  # Default sample rate

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()

        # Parsha selection
        parsha_layout = QHBoxLayout()
        parsha_label = QLabel("Parsha:")
        self.parsha_combo = QComboBox()
        self.parsha_combo.addItems(parsha_names)
        parsha_layout.addWidget(parsha_label)
        parsha_layout.addWidget(self.parsha_combo)

        # Reading selection
        reading_layout = QHBoxLayout()
        reading_label = QLabel("Reading:")
        self.reading_combo = QComboBox()
        self.reading_combo.addItems([str(i) for i in range(1, 8)])
        self.reading_combo.addItem("All")
        reading_layout.addWidget(reading_label)
        reading_layout.addWidget(self.reading_combo)

        # Load Parsha text button
        self.load_parsha_button = QPushButton("Load Parsha Text")
        self.load_parsha_button.clicked.connect(self.load_selected_parsha_text)

        layout.addLayout(parsha_layout)
        layout.addLayout(reading_layout)
        layout.addWidget(self.load_parsha_button)

        # YouTube link input
        youtube_layout = QHBoxLayout()
        youtube_label = QLabel("YouTube Link:")
        self.youtube_input = QLineEdit()
        youtube_button = QPushButton("Download")
        youtube_button.clicked.connect(self.download_youtube_audio)
        youtube_layout.addWidget(youtube_label)
        youtube_layout.addWidget(self.youtube_input)
        youtube_layout.addWidget(youtube_button)
        layout.addLayout(youtube_layout)

        # Full text box
        self.full_text = QTextEdit()
        self.full_text.setPlaceholderText("Enter full text here...")
        layout.addWidget(self.full_text)

        # Current segment text box
        self.segment_text = QTextEdit()
        self.segment_text.setPlaceholderText("Enter text for current segment...")
        layout.addWidget(self.segment_text)

        # Timeline
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setEnabled(False)
        self.timeline.sliderMoved.connect(self.seek_audio)
        self.timeline.sliderPressed.connect(self.timeline_pressed)
        self.timeline.sliderReleased.connect(self.timeline_released)
        self.timeline.setTracking(False)  # Disable automatic seeking while dragging
        layout.addWidget(self.timeline)

        # Time labels
        time_layout = QHBoxLayout()
        self.current_time_label = QLabel("00:00")
        self.total_time_label = QLabel("00:00")
        time_layout.addWidget(self.current_time_label)
        time_layout.addStretch()
        time_layout.addWidget(self.total_time_label)
        layout.addLayout(time_layout)

        # Playback controls
        playback_layout = QHBoxLayout()
        self.play_pause_button = QPushButton("Play")
        self.play_pause_button.clicked.connect(self.toggle_playback)
        playback_layout.addWidget(self.play_pause_button)
        layout.addLayout(playback_layout)

        # Other buttons
        button_layout = QHBoxLayout()
        self.load_button = QPushButton("Load Audio")
        self.load_button.clicked.connect(self.load_audio)
        button_layout.addWidget(self.load_button)

        self.next_button = QPushButton("Next Segment")
        self.next_button.clicked.connect(self.next_segment)
        button_layout.addWidget(self.next_button)

        self.save_button = QPushButton("Save Project")
        self.save_button.clicked.connect(self.save_project)
        button_layout.addWidget(self.save_button)

        self.load_project_button = QPushButton("Load Project")
        self.load_project_button.clicked.connect(self.load_project)
        button_layout.addWidget(self.load_project_button)

        layout.addLayout(button_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        central_widget.setLayout(layout)

        # Media player for audio playback
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.positionChanged.connect(self.update_timeline)
        self.media_player.durationChanged.connect(self.set_timeline_duration)

        # Timer for updating current time label
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_current_time)
        self.timer.start(100)  # Update every 100ms for smoother updates


    def show_error(self, title, message):
        """Shows an error message with a copyable text area."""
        error_box = QMessageBox()
        error_box.setIcon(QMessageBox.Icon.Critical)
        error_box.setWindowTitle(title)
        error_box.setText(message)

        text_edit = QPlainTextEdit()
        text_edit.setPlainText(message)
        text_edit.setReadOnly(True)
        error_box.layout().addWidget(text_edit)

        error_box.exec()
 


    def download_youtube_audio(self):
        youtube_link = self.youtube_input.text()
        if not youtube_link:
            self.show_error("Error", "Please enter a YouTube link.")
            return

        try:
            yt = pytube.YouTube(youtube_link)
            audio_stream = yt.streams.filter(only_audio=True).first()

            # Filter out invalid characters from the title
            safe_title = re.sub(r'[\\/*?:"<>|]', "", yt.title) 
            self.audio_file = audio_stream.download(filename=f"{safe_title}.mp3")
            self.output_dir = os.path.dirname(self.audio_file)
            self.load_selected_parsha_text()
            self.segment_audio()
        except Exception as e:
            self.show_error("Error", f"Failed to download YouTube audio: {e}")


    def load_audio(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Audio File", "", "Audio Files (*.wav *.mp3)")
        if file_name:
            try:
                self.audio_file = file_name
                self.output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
                if not self.output_dir:
                    return

                parsha_name = self.parsha_combo.currentText()
                reading = self.reading_combo.currentText()
                self.load_parsha_text(parsha_name, reading)  # Load Parsha text after loading audio
                self.segment_audio()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load audio file: {str(e)}")

    def load_selected_parsha_text(self):
        parsha_name = self.parsha_combo.currentText()
        reading = self.reading_combo.currentText()
        self.load_parsha_text(parsha_name, reading)

    def load_parsha_text(self, parsha_name, reading):
        if reading == "All":
            text = ""
            for i in range(1, 8):
                file_path = os.path.join("text", f"{parsha_name}-{i}.txt")
                if os.path.exists(file_path):
                    with open(file_path, "r", encoding="utf-8") as f:
                        text += f.read() + " "  # Add a space after each reading
        else:
            file_path = os.path.join("text", f"{parsha_name}-{reading}.txt")
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        
        # Remove newlines and extra spaces
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        self.full_text.setPlainText(text)


    def segment_audio(self):
        self.progress_bar.setVisible(True)
        self.load_button.setEnabled(False)

        self.processing_thread = QThread()
        self.worker = AudioProcessingWorker(
            self.audio_file,
            self.sr,
            max_segment_length=30,  # 30 seconds
            output_dir=self.output_dir
        )
        self.worker.moveToThread(self.processing_thread)
        self.processing_thread.started.connect(self.worker.process_audio)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.processing_finished.connect(self.finalize_segmentation)
        self.worker.processing_finished.connect(self.processing_thread.quit)
        self.processing_thread.finished.connect(self.worker.deleteLater)
        self.processing_thread.finished.connect(self.processing_thread.deleteLater)
        self.processing_thread.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def finalize_segmentation(self, segment_files, total_chunks):
        self.segment_files = segment_files
        self.current_segment = 0
        self.update_segment()
        self.progress_bar.setVisible(False)
        self.load_button.setEnabled(True)
        QMessageBox.information(self, "Success", f"Loaded audio file with {len(self.segment_files)} segments\nTotal chunks created: {total_chunks}")

    def play_current_segment(self):
        if self.current_segment < len(self.segment_files):
            segment_file = os.path.join(self.output_dir, self.segment_files[self.current_segment])
            self.media_player.setSource(QUrl.fromLocalFile(segment_file))
            self.media_player.play()

    def save_current_segment_text(self):
        current_text = self.segment_text.toPlainText()
        current_text = current_text.replace(" ׀ ", "׀").replace(" ׀ ", "׀").replace("׀", "׀ ").replace("־", "־ ").replace("[1]", "")
        current_text = re.sub(r'\s+|\n', ' ', current_text)  # replace multiple spaces or newline with a single space
        current_text = re.sub(r'\s+([.,;:!?])', r'\1', current_text) # remove space before punctuation
        current_text = re.sub(r'([.,;:!?])\s+', r'\1 ', current_text) # remove space after punctuation
        
        if self.current_segment < len(self.project_data["text"]):
            self.project_data["text"][self.current_segment] = current_text
        else:
            self.project_data["text"].append(current_text)

    def next_segment(self):
        if self.current_segment < len(self.segment_files) - 1:
            self.save_current_segment_text()
            self.current_segment += 1
            self.update_segment()
        else:
            self.media_player.stop()
            self.is_playing = False
            self.update_play_pause_button()
            QMessageBox.information(self, "End of Audio", "You've reached the end of the audio file.")


    def update_segment(self):
        if self.current_segment < len(self.segment_files):
            print(1)
            segment_file = os.path.join(self.output_dir, self.segment_files[self.current_segment])
            print(2)
            self.media_player.setSource(QUrl.fromLocalFile(segment_file))
            print(3)
            self.update_play_pause_button()
            print(4)
            self.timeline.setEnabled(True)
            
            # Get the duration of the current segment
            audio, _ = librosa.load(segment_file, sr=self.sr)
            self.segment_duration = len(audio) / self.sr * 1000  # Convert to milliseconds

            # Set the timeline duration *before* starting playback
            self.set_timeline_duration(int(self.segment_duration))

            self.play_current_segment()

            if self.current_segment < len(self.project_data["text"]):
                self.segment_text.setPlainText(self.project_data["text"][self.current_segment])
            else:
                self.segment_text.clear()
        else:
            self.media_player.stop()
            self.update_play_pause_button()
            self.timeline.setEnabled(False)
            QMessageBox.information(self, "End of Audio", "You've reached the end of the audio file.")

    def toggle_playback(self):
        if self.is_playing:
            self.media_player.pause()
            self.is_playing = False
        else:
            self.media_player.play()
            self.is_playing = True
        self.update_play_pause_button()

    def save_project(self):
        if not self.audio_file:
            QMessageBox.warning(self, "Error", "No audio file loaded.")
            return

        self.save_current_segment_text()

        self.project_data["audio"] = self.segment_files

        json_file = os.path.join(self.output_dir, "project_data.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.project_data, f, indent=2)

        QMessageBox.information(self, "Success", f"Project saved successfully.\nAudio segments and JSON file saved in: {self.output_dir}")

    def timeline_pressed(self): 
        if self.media_player.isPlaying():
            self.media_player.pause()

    def timeline_released(self):
        position = self.timeline.value()
        self.seek_audio(position)
        if self.is_playing:
            self.media_player.play()

    def seek_audio(self, position): 
        self.media_player.setPosition(position)
        self.update_current_time()

    def set_timeline_duration(self, duration):
        self.timeline.setRange(0, duration)
        self.update_total_time(duration)

    def update_current_time(self):
        position = self.media_player.position()
        self.current_time_label.setText(self.format_time(position))

    def update_total_time(self, duration):
        self.total_time_label.setText(self.format_time(duration))

    def format_time(self, ms):
        s = ms // 1000
        m, s = divmod(s, 60)
        return f"{m:02d}:{s:02d}"

    def closeEvent(self, event):
        # Ensure the processing thread is properly terminated
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.quit()
            self.processing_thread.wait()
        event.accept()

    def load_project(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Project", "", "JSON Files (*.json)")
        if file_name:
            with open(file_name, 'r', encoding='utf-8') as f:
                self.project_data = json.load(f)
            
            self.output_dir = os.path.dirname(file_name)
            self.segment_files = self.project_data["audio"]
            self.current_segment = 0
            self.update_segment()
            QMessageBox.information(self, "Success", "Project loaded successfully.")


    def update_play_pause_button(self):
        self.play_pause_button.setText("Pause" if self.is_playing else "Play")

    def update_timeline(self, position):
        self.timeline.setValue(position)
        self.update_current_time()

    def timeline_clicked(self, event):
        """Handles mouse click events on the timeline."""
        if event.button() == Qt.MouseButton.LeftButton:  # Check for left mouse button click
            if self.media_player.duration() > 0:
                timeline_width = self.timeline.width()
                click_position = event.pos().x()
                ratio = click_position / timeline_width
                new_position = int(ratio * self.media_player.duration())
                self.seek_audio(new_position)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioSegmentationApp()
    window.show()
    sys.exit(app.exec())

