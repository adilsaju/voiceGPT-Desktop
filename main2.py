import sys
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout
from PyQt6.QtGui import QFont, QPalette
from PyQt6.QtCore import Qt

class MyApp(QWidget):
    def __init__(self):
        super().__init__()

        # Set the window title
        self.setWindowTitle("voiceGPT")

        # Create a label for the text input
        input_label = QLabel("voiceGPT:")
        input_label.setFont(QFont("Arial", 12))

        # Create a text input
        self.input_text = QLineEdit()
        self.input_text.setFont(QFont("Arial", 12))

        # Create a red round button
        self.red_button = QPushButton("Click Me")
        self.red_button.setFixedSize(100, 100)
        self.red_button.setStyleSheet("background-color: red; border-radius: 50px;")

        # Create a red round button
        self.red_button2 = QPushButton("Click Me")
        self.red_button2.setFixedSize(100, 100)
        self.red_button2.setStyleSheet("background-color: orange; border-radius: 50px;")

        # Create a text box to display the result
        self.output_text = QLabel("")
        self.output_text.setFont(QFont("Arial", 12))

        # Set the layout for the app
        layout = QVBoxLayout()
        layout.addWidget(input_label)
        layout.addWidget(self.input_text)
        layout.addWidget(self.red_button)
        layout.addWidget(self.red_button2)

        layout.addWidget(self.output_text)
        self.setLayout(layout)

        # Connect the red button to the display_text method
        self.red_button.clicked.connect(self.start_record)

    def display_text(self):
        # Display the text from the text input in the text box
        self.output_text.setText(self.input_text.text())

    def start_record(self):
        None

    def stop_record(self):
        None
    

# Create the QApplication
app = QApplication(sys.argv)

# Create an instance of MyApp
window = MyApp()

# Show the window
window.show()

# Run the app
sys.exit(app.exec())
