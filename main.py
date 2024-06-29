import datetime
import os
import random
import sqlite3
import tkinter as tk
from tkinter.ttk import Progressbar
from time import sleep
from keras import callbacks, layers, models, optimizers
import mido
import music21
import numpy as np
from pygame import mixer


# class to train the machine learning model
class MelodyTrainer:
    def __init__(self, learning_rate, epochs, batch_size):
        self.data = None
        self.artist_for_training = "chopin"
        # This is the directory on my device where the classical music for training is stored
        self.root_directory = r"C:\Users\micha\Downloads\ClassicalMelodies"
        self.features = 128
        self._x_train = []
        self._y_train = []
        self.hidden_size = 256
        self.dropout = 0.3
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights_file = "chopin_weights.h5"
        self.model = None

    def load_data(self):
        # loads data from artists for training
        self.data = []

        for file in os.listdir(os.path.join(self.root_directory, self.artist_for_training)):
            # joins the root directory with the artist to load the correct music
            mid = mido.MidiFile(os.path.join(self.root_directory, self.artist_for_training, file))
            for msg in mid:
                if msg.type == "note_on":
                    # only takes notes that are played
                    ## Complex Feature - List Operations ##
                    self.data.append(msg.note)

    def create_sequences(self, sequence_length=100):
        # create input sequences and the corresponding outputs
        self._x_train = []
        self._y_train = []
        for i in range(0, len(self.data) - sequence_length, 1):
            # each x_train has length sequence length and y_train then has length 1 for the correct prediction
            self._x_train.append(self.data[i:i + sequence_length])
            # y_train is the next note after the sequence
            self._y_train.append(self.data[i + sequence_length])

        # converts the input data into real numbers between 0 and 1
        self._x_train = np.reshape(self._x_train, (len(self._x_train), sequence_length, 1))
        self._x_train = self._x_train / float(self.features)
        # converts the output data into a one-hot encoding
        self._y_train = array_to_one_hot(self._y_train, features=self.features)

    def configure_model(self):
        # defines the model's architecture
        self.model = models.Sequential()
        # defines first LSTM layer here
        self.model.add(layers.LSTM(
            self.hidden_size,
            input_shape=(self._x_train.shape[1], self._x_train.shape[2]),
            recurrent_dropout=self.dropout,
            return_sequences=True
        ))
        # defines hidden layer here
        self.model.add(layers.LSTM(self.hidden_size))
        # defines dense layer here
        self.model.add(layers.Dense(self.features))
        # defines activation layer here with activation function softmax
        self.model.add(layers.Activation("softmax"))
        # defines optimizer here with loss function categorical cross entropy and learning rate
        optimizer = optimizers.RMSprop(learning_rate=self.learning_rate)
        self.model.compile(loss="categorical_crossentropy", optimizer=optimizer)

    def train_model(self):
        # runs the model for the specified number of epochs
        # callbacks means that every epoch if the loss has decreased the weights will be saved
        self.model.fit(self._x_train, self._y_train, batch_size=self.batch_size, epochs=self.epochs,
                       callbacks=[callbacks.ModelCheckpoint(self.weights_file, monitor="loss", save_best_only=True)],
                       use_multiprocessing=True)

    def load_weights_for_training(self):
        # loads the previously trained weights, so it doesn't have to start from scratch
        self.model = models.load_model("chopin_weights.h5")


# class for the machine learning model
# inherits the MelodyTrainer class
class MelodyGenerator:
    def __init__(self, learning_rate, epochs, batch_size):
        # composition of MelodyTrainer class
        self.trainer = MelodyTrainer(learning_rate, epochs, batch_size)

        # model related variables
        self._lstm_weights = ((), (), ())
        self._dense_weights = ((), ())
        self._current_state = np.random.uniform(-0.1, 0.1, self.trainer.hidden_size)
        self._current_state = np.reshape(self._current_state, (1, self.trainer.hidden_size))
        self._hidden_state = np.random.uniform(-0.1, 0.1, self.trainer.hidden_size)
        self._hidden_state = np.reshape(self._hidden_state, (1, self.trainer.hidden_size))
        self.learning_rate = learning_rate
        self.output = None

        # music generation related variables
        self.name = None
        self.artist = None
        self.duration = None
        self.tempo = None
        self.instrument = None
        self.generated = False
        self.music_player = None

        # GUI related variables
        self.progress_report = None
        self.update_box = None
        self.root = None
        self._loaded_weights_for = None

    def load_weights_for_prediction(self):
        # loads the weights from the file to be used to generate music
        # must load it every time you generate as you can change the weights depending on the artist you pick
        self.trainer.model.load_weights(self.trainer.weights_file)
        weights = []
        # iterates through the layers of the model to get each set of weights
        for layer in self.trainer.model.layers:
            # no weights for the activation layer
            if layer.name != "activation":
                weights.append(layer.get_weights())
        return weights

    def _predict_output(self, previous):
        # given a previous note, predicts the next note
        # weights is organised as [lstm_weights1, lstm_weights2, dense_weights]
        weights = self.load_weights_for_prediction()
        # previous note passes into first lstm node
        self._lstm_weights = weights[0]
        self._lstm_layer(previous)
        # hidden state from last node is passed to the next layer
        self._lstm_weights = weights[1]
        self._lstm_layer(self._hidden_state)
        # hidden state from last node is passed into dense layer
        self._dense_weights = weights[2]
        dense_object = self._dense_layer()
        # dense reduces it to a single note output which is then passed into activation
        # output is currently one-hot encoded as this way it can get a probability model of each possible note
        output = activation_layer(dense_object, "softmax")
        # output is the predicted output
        return output

    def _lstm_layer(self, x):
        # LSTM represents an LSTM layer in the neural network and computes the feedforward calculations

        ## Complex Feature - Advanced Matrix Operation ##
        ## Complex Feature - Complex User-Defined Algorithm ##
        # the following code performs x * Wi + h * Ui + b
        # indentation across multiple lines is used to make it easier to read
        input_gate_intermediate = np.dot(x, self._lstm_weights[0][:, :self.trainer.hidden_size]) + \
                                  np.dot(self._hidden_state, self._lstm_weights[1][:, :self.trainer.hidden_size]) + \
                                  self._lstm_weights[2][:self.trainer.hidden_size]
        # the following code performs x * Wf + h * Uf + b
        forget_gate_intermediate = np.dot(x, self._lstm_weights[0][
                                             :, self.trainer.hidden_size:2 * self.trainer.hidden_size]) + \
                                   np.dot(self._hidden_state,
                                          self._lstm_weights[1][:, self.trainer.hidden_size:2 *
                                                                                            self.trainer.hidden_size]) \
                                   + self._lstm_weights[2][self.trainer.hidden_size:2 * self.trainer.hidden_size]
        # the following code performs x * Wc + h * Uc + b
        candidate_intermediate = np.dot(x, self._lstm_weights[0][
                                           :, 2 * self.trainer.hidden_size:3 * self.trainer.hidden_size]) + \
                                 np.dot(self._hidden_state,
                                        self._lstm_weights[1][:, 2 *
                                                                 self.trainer.hidden_size:3 *
                                                                                          self.trainer.hidden_size]) + \
                                 self._lstm_weights[2][2 * self.trainer.hidden_size:3 * self.trainer.hidden_size]
        # the following code performs x * Wo + h * Uo + b
        output_gate_intermediate = np.dot(x, self._lstm_weights[0][:, 3 * self.trainer.hidden_size:]) + \
                                   np.dot(self._hidden_state, self._lstm_weights[1][
                                                              :, 3 * self.trainer.hidden_size:]) + \
                                   self._lstm_weights[2][3 * self.trainer.hidden_size:]

        # the following code performs the activation functions on the intermediate values
        input_gate = sigmoid(input_gate_intermediate)  # first quarter
        forget_gate = sigmoid(forget_gate_intermediate)  # second quarter
        candidate = tanh(candidate_intermediate)  # third quarter
        output_gate = sigmoid(output_gate_intermediate)  # fourth quarter
        # np.multiply is element wise multiplication whereas np.dot is matrix multiplication
        self._current_state = np.multiply(input_gate, candidate) + np.multiply(forget_gate, self._current_state)
        self._hidden_state = np.multiply(output_gate, tanh(self._current_state))

    def _dense_layer(self):
        # dense layer is a simple matrix multiplication with a bias that returns a single note
        weight, bias = self._dense_weights
        dense_object = np.dot(self._hidden_state, weight) + bias
        return dense_object

    def generate_melody(self):
        # generates a melody using the trained model

        # first note is random confined between a pitch of 20 and 108 to ensure a reasonable note is chosen
        previous = [np.random.randint(20, self.trainer.features - 20) / self.trainer.features]

        output = []
        for note in range(self.duration):
            # each iteration you use the last note + states (memory) to predict the next
            one_hot_encoded = self._predict_output(np.array(previous))
            # gets a single note from the one-hot encoded output
            note = one_hot_to_integer(one_hot_encoded[0])
            output.append(note)
            # replaces the previous note with the new note
            previous = [note / self.trainer.features]
        return output

    def generate_melody_for_screen(self):
        # controls the process that happens when the user selects generate melody in the main window

        # check if all the options are selected
        if not self.validate_options():
            print("not validating")
            return None

        # updates screen
        self.update_progress_report("Generating Melody ...")

        # load relevant weights
        if not self.artist == self._loaded_weights_for:
            self.load_weights_for_prediction()
            self._loaded_weights_for = self.artist

        # generates melody
        self.output = self.generate_melody()
        # updates screen with name of song
        self.name = random_name_generator()
        self.update_progress_report(self.name)
        # allows music to be played
        array_to_midi(self.output, instrument=self.instrument, tempo=self.tempo)

        # this variable is important because when the user goes back to the main menu, if any music was loaded then that
        # music will be played regardless if any new music was generated
        self.generated = True
        # Must ensure the melody doesn't think it's paused otherwise it won't be able to play the new melody
        self.music_player.is_paused = False

    def validate_options(self):
        # checks if any of the options are not selected
        if None in (self.artist, self.instrument, self.tempo, self.duration):
            self.update_progress_report("Please select all options above")
            return False
        return True

    def reset_options(self):
        # resets all options to default
        self.artist = None
        self.instrument = None
        self.tempo = None
        self.duration = None

    def update_progress_report(self, text):
        # this updates the text in the box that tells the user what is happening
        self.progress_report.set(text)
        self.root.update()

    def save_melody(self, database):
        # if an output has been generated save file to database
        if self.output is not None:
            file = {"name": self.name, "dateCreated": datetime.datetime.now(), "artist": self.artist,
                    "instrument": self.instrument, "tempo": self.tempo, "duration": self.duration, "rating": None,
                    "notes": self.output}

            database.update_melodies_database(file, melody_id=None, get_max_id=True)
            self.update_progress_report("Melody Saved")
            # half-second sleep so that the user can read the text and know it has been saved
            # note that: it doesn't stop the user from being able to interact with the screen
            sleep(0.5)
            self.update_progress_report(self.name)


# class for database related functions
class Database:
    def __init__(self):
        self._database_file = "melodies.db"
        self.tkinter_database = None
        self.window_class = None

    def create_melodies_database(self):
        # creates database if it doesn't exist
        # uses with so that the connection is automatically closed to prevent file issues from occurring
        with sqlite3.connect(self._database_file) as connection:
            ## Complex Feature - User/CASE-generated DDL script ##
            cursor = connection.cursor()
            # creates melodies table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS melodies(
                MelodyID INTEGER PRIMARY KEY AUTOINCREMENT,
                Name STRING NOT NULL,
                DateCreated DATETIME NOT NULL,
                Instrument STRING NOT NULL,
                Duration INTEGER NOT NULL,
                Tempo INTEGER NOT NULL,
                Rating INTEGER
            ) """)
            # creates notes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS notes(
                NoteID INTEGER NOT NULL,
                MelodyID INTEGER NOT NULL,
                Notes INTEGER NOT NULL,
                PRIMARY KEY (NoteID, MelodyID),
                FOREIGN KEY (MelodyID) REFERENCES melodies(MelodyID)
            ) """)
            connection.commit()

    def update_melodies_database(self, file, melody_id, get_max_id=False):
        # writes a melody to the database if it doesn't match the file
        with sqlite3.connect(self._database_file) as connection:
            cursor = connection.cursor()
            # this means the id is unknown so the next available id is used
            if get_max_id:
                melody_id = self.get_next_melody_id()

            # if melody id not already in database
            if not cursor.execute("SELECT MelodyID FROM melodies WHERE MelodyID = ?", (melody_id,)).fetchone():
                ## Complex Feature - Cross-table parameterised SQL ##
                cursor.execute(
                    "INSERT INTO melodies (MelodyID, Name, DateCreated, Instrument, Duration, Tempo, Rating) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (melody_id, file["name"], file["dateCreated"], file["instrument"], file["duration"],
                     file["tempo"], file["rating"]))

                # insert the notes one by one into the notes table
                for noteID, note in enumerate(file["notes"]):
                    cursor.execute(
                        "INSERT INTO notes (NoteID, MelodyID, Notes) VALUES (?, ?, ?)", (noteID, melody_id, int(note)))
                connection.commit()

    def update_dictionary(self, tkinter_database):
        # updates the tkinter database if there are any new melodies in the database
        # tkinter database is a dictionary with key = melodyID and value = nested dictionary
        # tkinter database contains the melodies to be shown to the user in the database window
        # updates the tkinter database
        with sqlite3.connect(self._database_file) as connection:
            cursor = connection.cursor()

            cursor.execute("SELECT MelodyID FROM melodies")
            IDs = cursor.fetchall()
            # add any songs in database not already in tkinter database to tkinter database
            for melody_id in IDs:
                if melody_id not in tkinter_database:
                    # if the melody is not in the tkinter database
                    cursor.execute("SELECT * FROM melodies WHERE MelodyID = ?", melody_id)
                    details = cursor.fetchall()[0]

                    tkinter_database[details[0]] = {"Name": details[1], "DateCreated": details[2],
                                                    "Instrument": details[3],
                                                    "Duration": details[4], "Tempo": details[5], "Rating": details[6]}

            # any melodies in tkinter database not in database are deleted
            # create a copy of the dictionary keys
            keys = list(tkinter_database.keys())
            for melody_id in keys:
                if (melody_id,) not in IDs:
                    # delete the entry from the original dictionary
                    del tkinter_database[melody_id]

        return tkinter_database

    def get_next_melody_id(self):
        # function determines an ID to use for the next melody
        with sqlite3.connect(self._database_file) as connection:
            cursor = connection.cursor()
            ## Complex Feature - Aggregate SQL Function ##
            try:  # if melodies exist find the largest ID and add 1
                return cursor.execute("SELECT MAX(MelodyID) FROM melodies").fetchone()[0] + 1
            except TypeError:  # if no melodies exist
                return 1

    def delete_id(self, melody_id):
        # deletes a melody from the database with a given ID
        with sqlite3.connect(self._database_file) as connection:
            cursor = connection.cursor()
            # deletes the melody with a specific melody id from the database
            cursor.execute("DELETE FROM melodies WHERE MelodyID = ?", (melody_id,))
            cursor.execute("DELETE FROM notes WHERE MelodyID = ?", (melody_id,))
            # shifts all the melody ids down by 1 so that there are no gaps
            cursor.execute("UPDATE melodies SET MelodyID = MelodyID - 1 WHERE MelodyID > ?", (melody_id,))
            cursor.execute("UPDATE notes SET MelodyID = MelodyID - 1 WHERE MelodyID > ?", (melody_id,))
            connection.commit()

    def rate_melody(self, rating):
        # updates the rating of a melody in the database

        # get_relative_melody_position() determines what quadrant of the screen the mouse is in: 1, 2, 3, 4
        # top left is 1, top right is 2, bottom left is 3, bottom right is 4
        relative_melody_position = get_relative_melody_position(self.window_class.mouse_position)
        # determines how the melodies on the screen is sorted to determine what the melody id the user chose is
        melody_id = self.window_class.get_melody_from_position(relative_melody_position)
        with sqlite3.connect("melodies.db") as connection:
            cursor = connection.cursor()
            # updates the rating of the given melody id
            cursor.execute("UPDATE melodies SET Rating = ? WHERE MelodyID = ?", (rating, melody_id))
            connection.commit()
        # resets the window so the changes take effect
        self.window_class.change_window()


# class to control playing music
class MusicPlayer:
    def __init__(self):
        # file related variables
        self._midi_file = r"loaded_melody.mid"

        # tkinter related variables
        self.window_class = None

        # other variables
        self.is_playing = False
        self.is_paused = False
        self.mel_id = None

        mixer.init()

    def get_melody(self):
        # as above, get_relative_melody_position() determines what quadrant of the screen the mouse is in: 1, 2, 3, 4
        # top left is 1, top right is 2, bottom left is 3, bottom right is 4
        relative_melody_position = get_relative_melody_position(self.window_class.mouse_position)
        # determines how the melodies on the screen is sorted to determine what the melody id the user chose is
        melody_id = self.window_class.get_melody_from_position(relative_melody_position)

        # get notes from notes database with melody id
        with sqlite3.connect("melodies.db") as connection:
            cursor = connection.cursor()
            # get notes
            cursor.execute("SELECT Notes FROM notes WHERE MelodyID = ?", (melody_id,))
            notes = cursor.fetchall()  # fetchall as there are multiple notes instead of fetchone
            # convert to list
            notes = [note[0] for note in notes]
            # get instrument
            cursor.execute("SELECT Instrument FROM melodies WHERE MelodyID = ?", (melody_id,))
            instrument = cursor.fetchone()[0]
            # get tempo
            cursor.execute("SELECT Tempo FROM melodies WHERE MelodyID = ?", (melody_id,))
            tempo = cursor.fetchone()[0]

        # stores music in loaded_melody.mid
        array_to_midi(notes, self._midi_file, instrument, tempo)
        return melody_id

    def play_melody(self, melody_id):
        # gets song from database and plays it
        mixer.music.load(self._midi_file)
        mixer.music.play()
        self.mel_id = melody_id
        self.is_playing = True
        self.is_paused = False

    def unpause_melody(self, melody_id):
        # unpause the music
        mixer.music.unpause()
        self.mel_id = melody_id
        self.is_playing = True
        self.is_paused = False

    def get_and_play(self):
        # get melody loads the correct melody to the midi file, and then it is played
        melody_id = self.get_melody()
        self.play_melody(melody_id)

    def unpause_or_play(self, melody_id):
        # makes a decision based on whether there is music already loaded
        # if music is generated it decides what to do with it
        if self.window_class.melody_generator.generated:
            # determines if its paused or not and then plays from the start or unpauses the music
            if self.is_paused:
                self.unpause_melody(melody_id)
            else:
                self.play_melody(melody_id)

    def stop_melody(self, melody_id=None, need_id=True):
        # stops the melody from playing
        if need_id:
            if not melody_id:
                # as above, get_relative_melody_position() determines what quadrant of the screen the mouse is in:
                # 1, 2, 3, 4 top left is 1, top right is 2, bottom left is 3, bottom right is 4
                relative_melody_position = get_relative_melody_position(self.window_class.mouse_position)
                # determines how the melodies on the screen is sorted to determine what the melody id the user chose is
                melody_id = self.window_class.get_melody_from_position(relative_melody_position)

            # only if it's the correct pause button will it pause the melody otherwise it will stop the function
            if not (self.mel_id == melody_id and self.is_playing):
                return

        mixer.music.stop()
        self.is_playing = False

    def pause_melody(self):
        # pauses the melody if its playing
        if self.is_playing:
            mixer.music.pause()
            self.is_playing = False
            self.is_paused = True

    def get_midi_length(self):
        # returns the length of the midi file
        midi = mido.MidiFile(self._midi_file)
        length = midi.length
        return length


# parent class to window that creates widgets
class Widgets:
    def __init__(self, music_player):
        self.window = None
        self.music_player = music_player

    def add_text_box(self, text, position, font_size=10, bg_color="white", centred=True, font_type="Consolas"):
        # add a textbox to the window
        text_box = tk.Text(self.window, height=1, width=len(text), font=(font_type, font_size), bd=0, bg=bg_color)
        text_box.place(x=position[0], y=position[1])
        text_box.insert(tk.END, text)
        text_box.configure(state="disabled")  # makes it so the user can't edit the text

        if centred:
            text_box.tag_configure("centre", justify="center")
            text_box.tag_add("centre", "1.0", "end")
            text_box.pack()  # pack centres the widget
        return text_box  # function may not have a use for the text box, but it is returned just in case

    def add_label(self, text_var, font_size, position, bg_color="white", font_type="Consolas"):
        # add a label to the window
        label = tk.Label(self.window, textvariable=text_var, font=(font_type, font_size), bd=0, bg=bg_color)
        label.place(x=position[0], y=position[1])
        return label

    def add_box(self, position, size, colour):
        # add a box to the window
        canvas = tk.Canvas(self.window, width=size[0], height=size[1], bg=colour)
        canvas.place(x=position[0], y=position[1])
        return canvas

    def add_progress_bar(self, position, size):
        # add a progress bar to the window
        # determinate mode means that the progress bar will fill up as the music plays instead of a bar that travels
        # across the screen as the music is played
        progress_bar = Progressbar(self.window, orient="horizontal", length=size[0], mode="determinate")
        progress_bar.place(x=position[0], y=position[1])
        return progress_bar

    def add_button(self, text, position, size, command):
        # add a button to the window
        button = tk.Button(self.window, text=text, command=command)
        button.place(x=position[0], y=position[1], width=size[0], height=size[1])
        return button

    def add_dropdown_menu(self, choice, menu, position):
        # add a dropdown menu to the window
        dropdown = tk.OptionMenu(self.window, choice, *menu)
        dropdown.place(x=position[0], y=position[1])
        return dropdown

    def update_progress_bar(self, progress_bar):
        # updates the progress bar
        pos = mixer.music.get_pos() / 1000
        length = self.music_player.get_midi_length()

        percentage = pos / length * 100
        progress_bar["value"] = percentage
        # every 10 milliseconds the progress bar will check the position of the music and change the progress bar
        self.window.after(10, lambda: self.update_progress_bar(progress_bar))


# parent class to window, given a window widget it will create the main window
class MainWindow:
    def __init__(self, music_player):
        # composition of the widgets class
        self.widgets = Widgets(music_player)

        self.root = None
        self.melody_generator = None
        self.database = None

    @staticmethod
    def change_window(to_database=False):
        # method intended to be overridden by the window class to change the window
        print("Failed to change window as the database window has not been assigned")

    def _main_add_text(self):
        # adds text (not in the box) to the main window
        # supplementary function to main_window
        self.widgets.add_text_box("Classical Music Generator", font_size=50, position=(0, 1))
        self.widgets.add_text_box(
            "Simply customise the settings and click generate music to make music tailored to your tastes!",
            position=(1, 1))
        self.widgets.add_text_box(
            "Click generate music to get a new song, or click play/pause to play the current song.",
            position=(1, 2))
        self.widgets.add_text_box("Save the music to your database by clicking save!",
                                  position=(1, 3))
        self.widgets.add_text_box("In the style of", font_size=15, position=(100, 200), centred=False)
        self.widgets.add_text_box("Instrument", font_size=15, position=(500, 200), centred=False)
        self.widgets.add_text_box("Tempo", font_size=15, position=(100, 300), centred=False)
        self.widgets.add_text_box("Number of Notes", font_size=15, position=(500, 300), centred=False)

    def _main_add_box(self):
        # adds a box to the main window containing progress bar and buttons
        # supplementary function to main_window
        self.widgets.add_box(position=(100, 400), size=(800, 250), colour="gray")
        # tk.StringVar() means it may be changed in runtime
        self.melody_generator.progress_report = tk.StringVar()
        self.melody_generator.progress_report.set("No Melody Generated Yet")
        # this is the text box that will display the progress of the melody generator or the name of the music generated
        self.melody_generator.update_box = self.widgets.add_label(self.melody_generator.progress_report,
                                                                  position=(150, 420), bg_color="gray", font_size=20)

        progress_bar = self.widgets.add_progress_bar(position=(150, 500), size=(700, 50))
        self.widgets.add_button("Generate Music", position=(150, 580), size=(150, 50),
                                command=lambda: self.melody_generator.generate_melody_for_screen())
        self.widgets.add_button("Play Music", position=(350, 580), size=(150, 50),
                                command=lambda: self.widgets.music_player.unpause_or_play(
                                melody_id=self.database.get_next_melody_id()))
        self.widgets.add_button("Pause Music", position=(500, 580), size=(150, 50),
                                command=lambda: self.widgets.music_player.pause_melody())
        self.widgets.add_button("Save Music", position=(700, 580), size=(150, 50),
                                command=lambda: self.melody_generator.save_melody(self.database))
        self.widgets.add_button("Go To Database", position=(100, 653), size=(200, 50),
                                command=lambda: self.change_window(to_database=True))
        self.widgets.add_button("Close Window", position=(704, 653), size=(200, 50),
                                command=self._leave_program)
        self.widgets.update_progress_bar(progress_bar)

    def _main_add_menus(self):
        # adds choice dropdown menus to the main window
        # supplementary function to main_window
        Artists = ["Chopin", "Beethoven"]
        Instruments = ["Piano", "Guitar", "Violin", "Flute"]
        Tempos = ["60", "120", "240", "480"]
        Durations = ["25", "50", "75", "100"]

        # menu to choose artist style
        artist = tk.StringVar(self.widgets.window)
        artist.set("Pick an artist")
        self.widgets.add_dropdown_menu(artist, Artists, (350, 200))
        # menu to choose instrument
        instrument = tk.StringVar(self.widgets.window)
        instrument.set("Pick an instrument")
        self.widgets.add_dropdown_menu(instrument, Instruments, (750, 200))
        # menu to choose tempo
        tempo = tk.StringVar(self.widgets.window)
        tempo.set("Pick a tempo")
        self.widgets.add_dropdown_menu(tempo, Tempos, (350, 300))
        # menu to choose duration
        duration = tk.StringVar(self.widgets.window)
        duration.set("Pick a duration")
        self.widgets.add_dropdown_menu(duration, Durations, (750, 300))
        # trace the menus to track choices, setattr is used to update the attributes in the melody generator
        artist.trace("w", lambda *args: setattr(self.melody_generator, "artist", artist.get()))
        instrument.trace("w", lambda *args: setattr(self.melody_generator, "instrument", instrument.get()))
        tempo.trace("w", lambda *args: setattr(self.melody_generator, "tempo", int(tempo.get())))
        duration.trace("w", lambda *args: setattr(self.melody_generator, "duration", int(duration.get())))

    def main_window(self):
        # makes the main window
        self._main_add_text()
        self._main_add_box()
        self._main_add_menus()

    def _leave_program(self):
        # destroys the main window
        self.root.quit()
        # stops the program
        exit(0)


# parent class to window, given a window widget it will create the database window
class DatabaseWindow:
    def __init__(self, music_player):
        # composition of the widgets class
        self.widgets = Widgets(music_player)

        self.root = None
        self.database_sorting = "Name"
        self.database = None
        self.database_page = 0
        self.tkinter_database = {}
        self.mouse_position = (0, 0)

    @staticmethod
    def change_window(to_database=False):
        # method intended to be overridden by the window class to change the window
        print("Failed to change window as the database window has not been assigned")

    def reload_database(self, sorted_by=None):
        # reloads the database window to show any new changes
        if sorted_by:
            self.database_sorting = sorted_by
        self.change_window(to_database=True)

    def delete_melody(self):
        # deletes the melody from the database
        # as above, get_relative_melody_position() is used to get the quadrant the mouse is in: 1, 2, 3, 4
        # 1 is the top left, 2 is the top right, 3 is the bottom left and 4 is the bottom right
        relative_melody_position = get_relative_melody_position(self.mouse_position)
        melody_id = self.get_melody_from_position(relative_melody_position)

        self.database.delete_id(melody_id)
        # reloads window to show any changes
        self.reload_database()

    def _add_database_dropdown(self, menu, position, rating=False, sorting=False):
        # supplementary function to add a dropdown menu which helps make the database window
        # add a dropdown menu to the window
        # must be in this class and not widgets as it may be used to sort the screen which refreshes the screen and is
        # a local function
        choice = tk.StringVar(self.widgets.window)
        choice.set("Pick an option")
        self.widgets.add_dropdown_menu(choice, menu, position)
        # there are two dropdown menu options, it can either be used for rating a melody or sorting the melodies
        if rating:
            # choice.trace is used to detect when the user changes the rating of a melody
            # "w" refers to write, which means the callback function will be called when the user changes the rating
            choice.trace("w", lambda *args: self.database.rate_melody(int(choice.get())))
        elif sorting:
            choice.trace("w",
                         lambda *args: self.reload_database(choice.get()))

    def _database_add_text(self):
        # adds text to the database window
        # supplementary function to database_window
        self.widgets.add_text_box("Database", font_size=50, position=(0, 1))
        self.widgets.add_text_box("Here you can view and interact with the melodies that have been generated",
                                  font_size=10,
                                  position=(1, 1))

    def _database_add_sorting(self):
        # adds sorting options to the database window
        # supplementary function to database_window
        self.widgets.add_text_box("Sort By:", font_size=10, position=(825, 120), bg_color="white", centred=False)
        self._add_database_dropdown(["Name", "Duration", "Rating", "Date Created"], position=(825, 140),
                                    sorting=True)

    def _database_song_box(self, attributes, x, y):
        # adds a box about a melody and options to interact with it
        # supplementary function to database_window
        self.widgets.add_box((x - 10, y - 10), (310, 250), "gray")
        self.widgets.add_text_box("Name: " + attributes["Name"],
                                  font_size=15, position=(x, y), bg_color="gray", centred=False)
        self.widgets.add_text_box("Duration: " + str(attributes["Duration"]) + " notes",
                                  font_size=15, position=(x, y + 30), bg_color="gray", centred=False)
        # if the melody has been rated, display the rating, otherwise display "Not Rated"
        if attributes["Rating"]:
            self.widgets.add_text_box(f"Rating: {str(attributes['Rating'])}/5",
                                      font_size=15, position=(x, y + 60), bg_color="gray", centred=False)
        else:
            self.widgets.add_text_box("Rating: Not Rated",
                                      font_size=15, position=(x, y + 60), bg_color="gray", centred=False)

        # dropdown menu to allow the user to rate the melody
        self.widgets.add_text_box("Update Rating: ", font_size=15, position=(x, y + 90), bg_color="gray",
                                  centred=False)
        self._add_database_dropdown(["1", "2", "3", "4", "5"], position=(x + 160, y + 90),
                                    rating=True)
        # buttons to allow the user to play, stop or delete the melody
        self.widgets.add_button("Play Melody", position=(x, y + 150), size=(150, 40),
                                command=self.widgets.music_player.get_and_play)
        self.widgets.add_button("Stop Melody", position=(x + 150, y + 150), size=(150, 40),
                                command=self.widgets.music_player.stop_melody)
        self.widgets.add_button("Delete Melody", position=(x, y + 200), size=(150, 40),
                                command=self.delete_melody)

    def _database_iterate_widgets(self):
        # iterates through the melodies in the database and adds a box for each one
        # supplementary function to database_window
        counter = 0
        min_range = self.database_page * 4  # the first melody to be displayed on the screen
        max_range = min_range + 3  # the last melody to be displayed on the screen
        sorted_database = sort_dictionary(self.tkinter_database, self.database_sorting)

        for melody_id, attributes in sorted_database:
            if min_range <= counter <= max_range:
                relative_position = counter % 4  # the position of the melody in the page
                x = 100 if relative_position % 2 == 0 else 500  # if its even, it's left, else, it's right
                y = 150 if relative_position < 2 else 450  # if it's less than 2, it's top, else, it's bottom
                self._database_song_box(attributes, x, y)
            counter += 1

    def _database_change_page(self):
        # changes the page of the database
        # supplementary function to database_window
        if self.database_page > 0:
            self.widgets.add_button("⮜", position=(50, 700), size=(200, 50),
                                    command=lambda: self.next_database_page(-1))
        if self.database_page < (len(self.tkinter_database) - 1) // 4:
            self.widgets.add_button("⮞", position=(750, 700), size=(200, 50),
                                    command=lambda: self.next_database_page(1))

        self.widgets.add_button("Go To Main Menu", position=(50, 0), size=(200, 50),
                                command=lambda: self.change_window(to_database=False))
        self.widgets.add_button("Close Window", position=(750, 0), size=(200, 50),
                                command=self._leave_program)

    def database_window(self):
        # makes the database window
        self.widgets.window.bind("<Motion>", self._update_mouse_position)

        self._database_add_text()
        self._database_add_sorting()
        self._database_iterate_widgets()
        self._database_change_page()

    def _update_mouse_position(self, event):
        # updates the mouse position for determining what widgets the user is interacting with
        x, y = event.widget.winfo_x(), event.widget.winfo_y()
        self.mouse_position = (x, y)

    def next_database_page(self, change):
        # changes the page of the database. Change is either 1 or -1
        self.database_page += change
        self.reload_database()

    def get_melody_from_position(self, position):
        # gets the melody that the user is interacting with
        sorted_dictionary = sort_dictionary(self.tkinter_database, self.database_sorting)
        return sorted_dictionary[self.database_page * 4 + position - 1][0]

    def _leave_program(self):
        # destroys the window and quits the program
        self.root.quit()
        exit(0)


# class that creates the user interface and connects the main and database classes
class Window(MainWindow, DatabaseWindow):
    def __init__(self, melody_generator, music_player):
        # inherits all methods and attributes from main and database class
        MainWindow.__init__(self, music_player)
        DatabaseWindow.__init__(self, music_player)

        # tkinter related variables
        self.root = tk.Tk()
        self.root.withdraw()
        window = self.make_window("Melody Generator")
        self.widgets.window = window

        # generator related variables
        self.melody_generator = melody_generator
        self.melody_generator.root = self.root
        self.music_player = music_player
        self.melody_generator.music_player = self.music_player

    def make_window(self, title, size="1000x1000", resizable=False, bg_color="white"):
        # makes a new window
        window = tk.Toplevel(self.root)
        window.title(title)
        window.geometry(size)
        window.resizable(resizable, resizable)
        window.configure(background=bg_color)
        return window

    def change_window(self, to_database=True):
        # this code controls the change from one window to another

        # Check for any updates to the database
        self.tkinter_database = self.database.update_dictionary(self.tkinter_database)

        # melody updates
        # either window could be playing music that needs to be stopped
        self.music_player.stop_melody(need_id=False)

        if not to_database:
            # previously selected music settings need to be reset
            self.melody_generator.reset_options()
            self.melody_generator.generated = False

        self.widgets.window.destroy()
        # change the window
        if to_database:
            self.widgets.window = self.make_window("Database")
            self.database_window()
        else:
            self.widgets.window = self.make_window("Melody Generator")
            self.main_window()


def main():
    # Training the model - comment this line out if you don't want to train the model on running the program
    # train_model(25)

    # creates the melody generator
    network = create_network()

    # Creates the user interface
    create_interface(network)


def train_model(epochs):
    # trains the model
    print("Generating weights through training...")
    Trainer = MelodyTrainer(learning_rate=0.001, epochs=epochs, batch_size=64)
    Trainer.load_data()
    Trainer.create_sequences()
    Trainer.load_weights_for_training()
    Trainer.train_model()
    print("Training is complete.")


def create_network():
    # creates the melody generator
    print("Creating network to generate music...")
    network = MelodyGenerator(learning_rate=0.001, epochs=50, batch_size=64)
    network.trainer.load_data()
    network.trainer.create_sequences()
    network.trainer.configure_model()
    return network


def create_interface(network):
    # creates the user interface
    print("Creating User interface...")
    music_player = MusicPlayer()
    window_class = Window(network, music_player)
    database_class = Database()
    database_class.window_class = window_class
    window_class.database = database_class
    music_player.window_class = window_class
    window_class.main_window()
    window_class.root.mainloop()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1)


def relu(x):
    return np.maximum(0, x)


def activation_layer(x, activation):
    # in the case of this program, the activation is softmax, but I added relu for any future use since it's also a
    # common activation function
    activation_object = None
    if activation == "softmax":
        activation_object = softmax(x)
    elif activation == "relu":
        activation_object = relu(x)
    else:
        print("Invalid activation function.")
    # output note one-hot encoded
    return activation_object


def one_hot_to_integer(one_hot):
    # convert the one hot encoded note to an integer
    max_ind = 0
    max_val = 0
    for i in range(len(one_hot)):
        if one_hot[i] > max_val:
            max_val = one_hot[i]
            max_ind = i
    return max_ind + 1


def array_to_one_hot(arr, features=0):
    # converts an array of integers to a one hot encoded array
    # features is the number of features in the one hot encoded array
    # if features is None, then the maximum value in the array is used
    if not features:
        features = max(arr)

    new_arr = []
    for i in arr:
        one_hot = [0] * features
        one_hot[i - 1] = 1
        new_arr.append(one_hot)

    return np.array(new_arr)


def random_name_generator():
    # randomly picks two words to put together to create a random name
    words1 = ["Astral", "Celestial", "Cosmic", "Ethereal", "Galactic", "Heavenly", "Nebular", "Planetary", "Stellar"]
    words2 = ["Concerto", "Harmony", "Melody", "Overture", "Rhapsody", "Sonata", "Suite", "Symphony", "Toccata"]

    # 9 ^ 2 = 81 possible names
    name = random.choice(words1) + " " + random.choice(words2)
    return name


def get_relative_melody_position(mouse_position, x_size=1000, y_size=1000):
    # takes the mouse position and determines what quadrant of the screen the user is interacting with
    # currently, window size is 1000 x 1000, but if new functionality to change the window size were added, then
    # parameters must be changed
    x_position, y_position = mouse_position
    if 0 < x_position < x_size / 2 and 0 < y_position < y_size / 2:  # top left
        return 1
    elif 0 < x_position < x_size / 2 and y_size / 2 <= y_position < y_size:  # bottom left
        return 3
    elif x_size / 2 <= x_position < x_size and 0 < y_position < y_size / 2:  # top right
        return 2
    elif x_size / 2 <= x_position < x_size and y_size / 2 <= y_position < y_size:  # bottom right
        return 4
    else:
        # unknown quadrant
        return None


def array_to_midi(melody, file="loaded_melody.mid", instrument="Piano", tempo=120):
    # takes an array of notes and converts it to a midi file

    # create a new Music21 Stream object
    stream = music21.stream.Stream()

    # set the tempo of the stream
    stream.timeSignature = music21.meter.TimeSignature("4/4")

    # iterate over the notes in the array
    for note in melody:
        # create a new Music21 Note object for the current note
        n = music21.note.Note(note)
        # sets the duration of the music
        n.duration = music21.duration.Duration(120 / tempo)
        # sets the volume of the music
        n.volume = music21.volume.Volume(velocity=90)
        # add the note to the stream
        stream.append(n)
        # sets the instrument of the music
        stream.append(music21.instrument.fromString(instrument))

    # save the stream to a MIDI file
    # overwrites the file if it already exists
    stream.write("midi", file, overwrite=True)

    # gets rid of any music loaded in the mixer
    mixer.music.stop()


def sort_dictionary(dictionary, sorted_by="Name", error_check=True):
    # sorts a dictionary but for this program, used to sort tkinter database
    # removes spaces since all the keys in the dictionary have no spaces
    sorted_by = sorted_by.replace(" ", "")
    arr = [element for element in dictionary.items()]
    none_types = []
    if error_check:
        # specifically important for rating since you can have no rating but in general can be used for any situation
        arr, none_types = remove_none_types(arr, sorted_by)

    if sorted_by != "Name":
        sorted_database = timsort(arr, sorted_by, descending=True)
    else:
        # aesthetic choice to have the names sorted in alphabetically ascending order
        sorted_database = timsort(arr, sorted_by, descending=False)

    new_arr = sorted_database + none_types
    return new_arr


def merge(left, right, sorted_by, descending=True):
    # works in conjunction with timsort and performs a mergesort on the two arrays left and right
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        # sorts in descending order
        if not descending:
            # compares each element and adds the smaller of the two to the result
            if left[i][1][sorted_by] <= right[j][1][sorted_by]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        else:
            if left[i][1][sorted_by] >= right[j][1][sorted_by]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
    result += left[i:]
    result += right[j:]
    return result


def timsort(arr, sorted_by, descending):
    ## Complex Feature - TimSort ##
    ## Complex Feature - Recursion ##
    # note that this code works for strings as comparing two strings with a >, <, <=, >= operator will compare the
    # associated ascii values which are strings

    # note the required input for this arr is a list of tuples where the first element is the key and the second element
    # is the value. This is because you can't sort values in a dictionary as they are unordered.

    n = len(arr)
    if n <= 1:
        return arr

    mid = n // 2

    left = timsort([elem for elem in arr[:mid]], sorted_by, descending)
    right = timsort([elem for elem in arr[mid:]], sorted_by, descending)

    return merge(left, right, sorted_by, descending)


def remove_none_types(arr, sorted_by):
    # goes through items in arr and moves all the None types to a new array
    new_arr = []
    i = 0
    while i < len(arr):
        if arr[i][1][sorted_by] is None:
            new_arr.append(arr.pop(i))
            i -= 1
        i += 1
    return arr, new_arr


if __name__ == "__main__":
    main()
