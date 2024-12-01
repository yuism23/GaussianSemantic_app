# ui_main_window.py

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200, 800)  # 根据需要调整尺寸

        # 创建中央窗口部件
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # 设置主垂直布局
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")

        # 添加渲染图像显示标签
        self.imageLabel = QtWidgets.QLabel(self.centralwidget)
        self.imageLabel.setMinimumSize(QtCore.QSize(800, 600))
        self.imageLabel.setFrameShape(QtWidgets.QFrame.Box)
        self.imageLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.imageLabel.setObjectName("imageLabel")
        self.verticalLayout.addWidget(self.imageLabel)

        # 添加相机信息显示标签
        self.cameraInfoLabel = QtWidgets.QLabel(self.centralwidget)
        self.cameraInfoLabel.setMinimumSize(QtCore.QSize(800, 30))
        self.cameraInfoLabel.setFrameShape(QtWidgets.QFrame.Panel)
        self.cameraInfoLabel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.cameraInfoLabel.setObjectName("cameraInfoLabel")
        self.verticalLayout.addWidget(self.cameraInfoLabel)

        # 创建控制按钮和进度条的水平布局
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")

        # 加载点云按钮
        self.loadButton = QtWidgets.QPushButton(self.centralwidget)
        self.loadButton.setObjectName("loadButton")
        self.horizontalLayout.addWidget(self.loadButton)

        # 渲染按钮
        self.renderButton = QtWidgets.QPushButton(self.centralwidget)
        self.renderButton.setObjectName("renderButton")
        self.horizontalLayout.addWidget(self.renderButton)

        # 开始训练按钮
        self.trainButton = QtWidgets.QPushButton(self.centralwidget)
        self.trainButton.setObjectName("trainButton")
        self.horizontalLayout.addWidget(self.trainButton)

        # 添加进度条
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.horizontalLayout.addWidget(self.progressBar)

        # 将按钮和进度条的布局添加到主垂直布局中
        self.verticalLayout.addLayout(self.horizontalLayout)

        # 创建相机控制的水平布局
        self.cameraControlsLayout = QtWidgets.QHBoxLayout()
        self.cameraControlsLayout.setObjectName("cameraControlsLayout")

        # 相机位置控件
        self.posXSpinBox = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.posXSpinBox.setPrefix("Pos X: ")
        self.posXSpinBox.setDecimals(2)
        self.posXSpinBox.setMinimum(-1000.0)
        self.posXSpinBox.setMaximum(1000.0)
        self.posXSpinBox.setSingleStep(0.1)
        self.posXSpinBox.setProperty("value", 0)
        self.posXSpinBox.setObjectName("posXSpinBox")
        self.cameraControlsLayout.addWidget(self.posXSpinBox)

        self.posYSpinBox = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.posYSpinBox.setPrefix("Pos Y: ")
        self.posYSpinBox.setDecimals(2)
        self.posYSpinBox.setMinimum(-1000.0)
        self.posYSpinBox.setMaximum(1000.0)
        self.posYSpinBox.setSingleStep(0.1)
        self.posYSpinBox.setProperty("value", 0)
        self.posYSpinBox.setObjectName("posYSpinBox")
        self.cameraControlsLayout.addWidget(self.posYSpinBox)

        self.posZSpinBox = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.posZSpinBox.setPrefix("Pos Z: ")
        self.posZSpinBox.setDecimals(2)
        self.posZSpinBox.setMinimum(-1000.0)
        self.posZSpinBox.setMaximum(1000.0)
        self.posZSpinBox.setSingleStep(0.1)
        self.posZSpinBox.setProperty("value", 5.0)
        self.posZSpinBox.setObjectName("posZSpinBox")
        self.cameraControlsLayout.addWidget(self.posZSpinBox)

        # 相机方向控件（四元数）
        self.orientWSpinBox = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.orientWSpinBox.setPrefix("Orient W: ")
        self.orientWSpinBox.setDecimals(4)
        self.orientWSpinBox.setMinimum(-1.0)
        self.orientWSpinBox.setMaximum(1.0)
        self.orientWSpinBox.setSingleStep(0.01)
        self.orientWSpinBox.setProperty("value", 1.0)
        self.orientWSpinBox.setObjectName("orientWSpinBox")
        self.cameraControlsLayout.addWidget(self.orientWSpinBox)

        self.orientXSpinBox = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.orientXSpinBox.setPrefix("Orient X: ")
        self.orientXSpinBox.setDecimals(4)
        self.orientXSpinBox.setMinimum(-1.0)
        self.orientXSpinBox.setMaximum(1.0)
        self.orientXSpinBox.setSingleStep(0.01)
        self.orientXSpinBox.setProperty("value", 0.0)
        self.orientXSpinBox.setObjectName("orientXSpinBox")
        self.cameraControlsLayout.addWidget(self.orientXSpinBox)

        self.orientYSpinBox = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.orientYSpinBox.setPrefix("Orient Y: ")
        self.orientYSpinBox.setDecimals(4)
        self.orientYSpinBox.setMinimum(-1.0)
        self.orientYSpinBox.setMaximum(1.0)
        self.orientYSpinBox.setSingleStep(0.01)
        self.orientYSpinBox.setProperty("value", 0.0)
        self.orientYSpinBox.setObjectName("orientYSpinBox")
        self.cameraControlsLayout.addWidget(self.orientYSpinBox)

        self.orientZSpinBox = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.orientZSpinBox.setPrefix("Orient Z: ")
        self.orientZSpinBox.setDecimals(4)
        self.orientZSpinBox.setMinimum(-1.0)
        self.orientZSpinBox.setMaximum(1.0)
        self.orientZSpinBox.setSingleStep(0.01)
        self.orientZSpinBox.setProperty("value", 0.0)
        self.orientZSpinBox.setObjectName("orientZSpinBox")
        self.cameraControlsLayout.addWidget(self.orientZSpinBox)

        # 将相机控制布局添加到主垂直布局中
        self.verticalLayout.addLayout(self.cameraControlsLayout)

        # 将中央窗口部件设置为主窗口的中央部件
        MainWindow.setCentralWidget(self.centralwidget)

        # 创建菜单栏（可选）
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1200, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        # 创建状态栏（可选）
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        # 设置标签的初始文本
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "GaussianSemantic_app"))
        self.imageLabel.setText(_translate("MainWindow", "Rendered Image Will Appear Here"))
        self.cameraInfoLabel.setText(_translate("MainWindow", "Position: (0.00, 0.00, 0.00) | Orientation: (1.00, 0.00, 0.00)"))
        self.loadButton.setText(_translate("MainWindow", "Load Point Cloud"))
        self.renderButton.setText(_translate("MainWindow", "Render"))
        self.trainButton.setText(_translate("MainWindow", "Start Training"))

   