# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QTableWidgetItem

from main import MainProgram
from classifiers import KNNClassifier, SklearnDecisionTree, NaiveBayesClassifier, SklearnRandomForest

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 723)
        MainWindow.setAcceptDrops(False)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(-1, -1, 801, 731))
        self.frame.setStyleSheet("background-color: rgb(239, 241, 255);\n"
"border-color: rgb(0, 0, 0);")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.tabWidget = QtWidgets.QTabWidget(self.frame)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 801, 731))
        self.tabWidget.setObjectName("tabWidget")
        self.classifiers_tab = QtWidgets.QWidget()
        self.classifiers_tab.setObjectName("classifiers_tab")
        self.naive_bayes_label = QtWidgets.QLabel(self.classifiers_tab)
        self.naive_bayes_label.setGeometry(QtCore.QRect(20, 80, 131, 51))
        self.naive_bayes_label.setObjectName("naive_bayes_label")
        self.knn_label = QtWidgets.QLabel(self.classifiers_tab)
        self.knn_label.setGeometry(QtCore.QRect(20, 150, 131, 51))
        self.knn_label.setObjectName("knn_label")
        self.naive_bayes_acc_label = QtWidgets.QLabel(self.classifiers_tab)
        self.naive_bayes_acc_label.setGeometry(QtCore.QRect(180, 80, 91, 51))
        self.naive_bayes_acc_label.setObjectName("naive_bayes_acc_label")
        self.knn_acc_label = QtWidgets.QLabel(self.classifiers_tab)
        self.knn_acc_label.setGeometry(QtCore.QRect(180, 150, 91, 51))
        self.knn_acc_label.setObjectName("knn_acc_label")
        self.custom_input_text_box = QtWidgets.QPlainTextEdit(self.classifiers_tab)
        self.custom_input_text_box.setGeometry(QtCore.QRect(20, 280, 741, 251))
        self.custom_input_text_box.setObjectName("custom_input_text_box")
        self.custom_input_label = QtWidgets.QLabel(self.classifiers_tab)
        self.custom_input_label.setGeometry(QtCore.QRect(20, 240, 131, 31))
        self.custom_input_label.setObjectName("custom_input_label")
        self.label_6 = QtWidgets.QLabel(self.classifiers_tab)
        self.label_6.setGeometry(QtCore.QRect(20, 10, 221, 31))
        self.label_6.setObjectName("label_6")
        
        self.button_select_csv = QtWidgets.QPushButton(self.classifiers_tab)
        self.button_select_csv.setGeometry(QtCore.QRect(680, -10, 111, 28))
        self.button_select_csv.setObjectName("button_select_csv")
        
        # Button event
        self.button_select_csv.clicked.connect(self.button_signal_select_csv)
        
        self.decision_tree_acc_label = QtWidgets.QLabel(self.classifiers_tab)
        self.decision_tree_acc_label.setGeometry(QtCore.QRect(610, 80, 91, 51))
        self.decision_tree_acc_label.setObjectName("decision_tree_acc_label")
        self.decision_tree_label = QtWidgets.QLabel(self.classifiers_tab)
        self.decision_tree_label.setGeometry(QtCore.QRect(430, 80, 151, 51))
        self.decision_tree_label.setObjectName("decision_tree_label")
        self.random_forrest_label = QtWidgets.QLabel(self.classifiers_tab)
        self.random_forrest_label.setGeometry(QtCore.QRect(430, 150, 151, 51))
        self.random_forrest_label.setObjectName("random_forrest_label")
        self.rf_acc_label = QtWidgets.QLabel(self.classifiers_tab)
        self.rf_acc_label.setGeometry(QtCore.QRect(610, 150, 91, 51))
        self.rf_acc_label.setObjectName("rf_acc_label")
        
        self.submit_custom_input = QtWidgets.QPushButton(self.classifiers_tab)
        self.submit_custom_input.setGeometry(QtCore.QRect(20, 540, 111, 28))
        self.submit_custom_input.setObjectName("submit_custom_input")
        
        # Button event
        self.submit_custom_input.clicked.connect(self.button_signal_submit_custom_input)
        
        self.nb_classify = QtWidgets.QPushButton(self.classifiers_tab)
        self.nb_classify.setGeometry(QtCore.QRect(280, 80, 71, 51))
        self.nb_classify.setObjectName("nb_classify")
        # Button event
        self.nb_classify.clicked.connect(self.naive_bayes_classify)
        
        self.knn_classify = QtWidgets.QPushButton(self.classifiers_tab)
        self.knn_classify.setGeometry(QtCore.QRect(280, 150, 71, 51))
        self.knn_classify.setObjectName("knn_classify")
         # Button event
        self.knn_classify.clicked.connect(self.k_nearst_neighbour_classify)
        
        self.decision_tree_classify = QtWidgets.QPushButton(self.classifiers_tab)
        self.decision_tree_classify.setGeometry(QtCore.QRect(720, 80, 71, 51))
        self.decision_tree_classify.setObjectName("decision_tree_classify")
        # Button event
        self.decision_tree_classify.clicked.connect(self.sklearn_decision_tree_classify)
        
        
        self.rf_classify = QtWidgets.QPushButton(self.classifiers_tab)
        self.rf_classify.setGeometry(QtCore.QRect(720, 150, 71, 51))
        self.rf_classify.setObjectName("rf_classify")
        # Button event
        self.rf_classify.clicked.connect(self.sklearn_rf_classify)
        
        
        self.label = QtWidgets.QLabel(self.classifiers_tab)
        self.label.setGeometry(QtCore.QRect(20, 60, 131, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.classifiers_tab)
        self.label_2.setGeometry(QtCore.QRect(170, 60, 91, 16))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.classifiers_tab)
        self.label_3.setGeometry(QtCore.QRect(420, 60, 131, 16))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.classifiers_tab)
        self.label_4.setGeometry(QtCore.QRect(600, 60, 91, 16))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.classifiers_tab)
        self.label_5.setGeometry(QtCore.QRect(40, 600, 141, 16))
        self.label_5.setObjectName("label_5")
        self.label_7 = QtWidgets.QLabel(self.classifiers_tab)
        self.label_7.setGeometry(QtCore.QRect(40, 630, 111, 16))
        self.label_7.setObjectName("label_7")
        self.custom_text_nb_label = QtWidgets.QLabel(self.classifiers_tab)
        self.custom_text_nb_label.setGeometry(QtCore.QRect(210, 600, 55, 16))
        self.custom_text_nb_label.setObjectName("custom_text_nb_label")
        self.custom_text_knn_label = QtWidgets.QLabel(self.classifiers_tab)
        self.custom_text_knn_label.setGeometry(QtCore.QRect(210, 630, 55, 16))
        self.custom_text_knn_label.setObjectName("custom_text_knn_label")
        self.label_10 = QtWidgets.QLabel(self.classifiers_tab)
        self.label_10.setGeometry(QtCore.QRect(350, 600, 141, 20))
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(self.classifiers_tab)
        self.label_11.setGeometry(QtCore.QRect(350, 630, 161, 20))
        self.label_11.setObjectName("label_11")
        self.custom_text_dt_label = QtWidgets.QLabel(self.classifiers_tab)
        self.custom_text_dt_label.setGeometry(QtCore.QRect(540, 600, 55, 16))
        self.custom_text_dt_label.setObjectName("custom_text_dt_label")
        self.custom_text_rf_label = QtWidgets.QLabel(self.classifiers_tab)
        self.custom_text_rf_label.setGeometry(QtCore.QRect(540, 630, 55, 16))
        self.custom_text_rf_label.setObjectName("custom_text_rf_label")
        self.tabWidget.addTab(self.classifiers_tab, "")
        self.data_analysis_tab = QtWidgets.QWidget()
        self.data_analysis_tab.setObjectName("data_analysis_tab")
        self.image_frame = QtWidgets.QFrame(self.data_analysis_tab)
        self.image_frame.setGeometry(QtCore.QRect(-10, 0, 821, 711))
        self.image_frame.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.image_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.image_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.image_frame.setObjectName("image_frame")
        self.comboBox = QtWidgets.QComboBox(self.image_frame)
        self.comboBox.setGeometry(QtCore.QRect(20, 10, 73, 22))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.pushButton = QtWidgets.QPushButton(self.image_frame)
        self.pushButton.setGeometry(QtCore.QRect(100, 10, 93, 21))
        self.pushButton.setObjectName("pushButton")
        # Button event
        self.pushButton.clicked.connect(self.fill_the_table)
        
        
        self.pushButton_2 = QtWidgets.QPushButton(self.image_frame)
        self.pushButton_2.setGeometry(QtCore.QRect(200, 10, 93, 21))
        self.pushButton_2.setObjectName("pushButton_2")
        # Button event
        self.pushButton_2.clicked.connect(self.plot_data)
        
    
        self.tableWidget = QtWidgets.QTableWidget(self.image_frame)
        self.tableWidget.setGeometry(QtCore.QRect(20, 40, 781, 651))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(1)
        self.tableWidget.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, item)
        self.tabWidget.addTab(self.data_analysis_tab, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.main = MainProgram()
        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
    # Signal Function for Select CSV button
    def button_signal_select_csv(self):
        filename = QFileDialog.getOpenFileName()
        path = filename[0]
        if path is not None:
            self.flag = 1
            self.pre_process(path)
    
    # Function that initiates pre-process
    def pre_process(self, path):
        self.df = self.main.read_csv(path)
        self.df =self.main.text_pre_process(self.df)
        self.word_dictionary = self.main.dictionary(self.df)
        X, y = self.main.vectorize(self.df, self.word_dictionary)
        self.X_train, self.X_test, self.y_train, self.y_test = self.main.test_train_split(X, y)
        
        
    def naive_bayes_classify(self):
        nb = NaiveBayesClassifier()
        nb.train(self.X_train, self.y_train)
        predictions = nb.predict(self.X_test)
        accuracy = str(self.main.accuracy(self.y_test, predictions))
        accuracy = accuracy[0:4]
        self.naive_bayes_acc_label.setText(accuracy)
        
        
    def k_nearst_neighbour_classify(self):    
        knn = KNNClassifier()
        predictions = knn.predict_classification(self.X_train, self.y_train, self.X_test)
        accuracy = str(self.main.accuracy(self.y_test, predictions))
        accuracy = accuracy[0:4]
        self.knn_acc_label.setText(accuracy)
    
    
    def sklearn_decision_tree_classify(self):    
        dt = SklearnDecisionTree()
        predictions = dt.decision_tree(self.X_train, self.y_train, self.X_test)
        accuracy = str(self.main.accuracy(self.y_test, predictions))
        accuracy = accuracy[0:4]
        self.decision_tree_acc_label.setText(accuracy)    
    
    
    def sklearn_rf_classify(self):    
        rf = SklearnRandomForest()
        predictions = rf.random_forest(self.X_train, self.y_train, self.X_test)
        accuracy = str(self.main.accuracy(self.y_test, predictions))
        accuracy = accuracy[0:4]
        self.rf_acc_label.setText(accuracy)
        
    def button_signal_submit_custom_input(self):
        self.custom_text = self.custom_input_text_box.toPlainText()
        if len(self.custom_text) > 10:
            custom_input_vector = self.main.custom_input(self.custom_text, self.word_dictionary)
            self.classify_custom_input(custom_input_vector)
     
        
    def classify_custom_input(self, custom_input_vector):
        nb = NaiveBayesClassifier()
        nb.train(self.X_train, self.y_train)
        prediction = nb.predict([custom_input_vector])
        self.custom_text_nb_label.setText(str(prediction[0]))
        
        knn = KNNClassifier()
        prediction = knn.predict_classification(self.X_train, self.y_train, [custom_input_vector])
        self.custom_text_knn_label.setText(str(prediction[0]))
        
        rf = SklearnRandomForest()
        prediction = rf.random_forest(self.X_train, self.y_train, [custom_input_vector])
        self.custom_text_dt_label.setText(str(prediction[0]))
        
        dt = SklearnDecisionTree()
        prediction = dt.decision_tree(self.X_train, self.y_train, [custom_input_vector])
        self.custom_text_rf_label.setText(str(prediction[0]))
        
        
    def fill_the_table(self):
        genre = str(self.comboBox.currentText())
        a, b = self.main.popular_words(self.df, genre)
        self.tableWidget.setRowCount(20)
        self.tableWidget.setVerticalHeaderLabels(a)
        for i in range(0,20):
            self.tableWidget.setItem(i,0, QTableWidgetItem(str(b[i])))
            
            
    def plot_data(self):
        genre = str(self.comboBox.currentText())
        self.main.plot_data(self.df, genre)
        

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "AI Final Project"))
        self.naive_bayes_label.setText(_translate("MainWindow", "Naive Bayes Classifier"))
        self.knn_label.setText(_translate("MainWindow", "KNN Classifier"))
        self.naive_bayes_acc_label.setText(_translate("MainWindow", "NA"))
        self.knn_acc_label.setText(_translate("MainWindow", "NA"))
        self.custom_input_label.setText(_translate("MainWindow", "Custom Input"))
        self.label_6.setText(_translate("MainWindow", "Select CSV File or enter custom input "))
        self.button_select_csv.setText(_translate("MainWindow", "Select CSV File"))
        self.decision_tree_acc_label.setText(_translate("MainWindow", "NA"))
        self.decision_tree_label.setText(_translate("MainWindow", "Decision Tree Classifier"))
        self.random_forrest_label.setText(_translate("MainWindow", "Random Forrest Classifier"))
        self.rf_acc_label.setText(_translate("MainWindow", "NA"))
        self.submit_custom_input.setText(_translate("MainWindow", "Submit"))
        self.nb_classify.setText(_translate("MainWindow", "Classify"))
        self.knn_classify.setText(_translate("MainWindow", "Classify"))
        self.decision_tree_classify.setText(_translate("MainWindow", "Classify"))
        self.rf_classify.setText(_translate("MainWindow", "Classify"))
        self.label.setText(_translate("MainWindow", "Classifier"))
        self.label_2.setText(_translate("MainWindow", "Accuracy"))
        self.label_3.setText(_translate("MainWindow", "Classifier"))
        self.label_4.setText(_translate("MainWindow", "Accuracy"))
        self.label_5.setText(_translate("MainWindow", "Naive Bayes Classifier :"))
        self.label_7.setText(_translate("MainWindow", "KNN Classifier :"))
        self.custom_text_nb_label.setText(_translate("MainWindow", "-"))
        self.custom_text_knn_label.setText(_translate("MainWindow", "-"))
        self.label_10.setText(_translate("MainWindow", "Decision Tree Classifier"))
        self.label_11.setText(_translate("MainWindow", "Random Forrest Classifier"))
        self.custom_text_dt_label.setText(_translate("MainWindow", "-"))
        self.custom_text_rf_label.setText(_translate("MainWindow", "-"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.classifiers_tab), _translate("MainWindow", "Classifiers"))
        self.comboBox.setItemText(0, _translate("MainWindow", "Action"))
        self.comboBox.setItemText(1, _translate("MainWindow", "Romance"))
        self.comboBox.setItemText(2, _translate("MainWindow", "Horror"))
        self.comboBox.setItemText(3, _translate("MainWindow", "Fantasy"))
        self.comboBox.setItemText(4, _translate("MainWindow", "Drama"))
        self.comboBox.setItemText(5, _translate("MainWindow", "Sci-Fi"))
        self.comboBox.setItemText(6, _translate("MainWindow", "Comedy"))
        self.comboBox.setItemText(7, _translate("MainWindow", "Crime"))
        self.pushButton.setText(_translate("MainWindow", "Fill the Table"))
        self.pushButton_2.setText(_translate("MainWindow", "Draw the Plot"))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Count"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.data_analysis_tab), _translate("MainWindow", "Data Analysis"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

