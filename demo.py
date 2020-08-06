from Utils.db_utils import *
from Utils.utils import *

from kivy.config import Config
Config.set('graphics', 'window_state', 'maximized')

from kivy.core.window import Window 
Window.clearcolor = (1, 1, 1, 1)

import kivy
from kivy.app import App
from kivy.uix.label import Label 
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.image import Image 

from kivy.clock import Clock
from kivy.uix.widget import Widget
from kivy.graphics.texture import Texture

from kivy.uix.screenmanager import ScreenManager, Screen

from kivy.graphics import Color, Rectangle

import os
import subprocess
import sys
import datetime
import cv2
 
kivy.require("1.11.1")


global usernames
global embeddings

global attendance 
global camera_index
global show_attendance_obj

camera_index = 0
image_enhance = 'off'
image_enhance_use_gpu = 'off'

attendance = dict()


class Home_Page(GridLayout):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.cols = 2
		self.padding = [100, 100, 100, 100]
		self.add_widget(Image(source = 'image.jpg'))

		buttons = GridLayout(cols = 2)

		buttons.spacing = [20, 20]

		button_1 = Button(text='BIOMETRIC \nVERIFICATION', font_size = 45, italic = True, background_color = [1, 255, 1, 1])
		button_2 = Button(text='SHOW \nATTENDANCE', font_size = 45, italic = True, background_color = [255, 1, 1, 1])
		button_3 = Button(text='USER \nREGISTRATION', font_size = 45, italic = True, background_color = [1, 1, 255, 1])
		button_4 = Button(text='USER \nDELETION', font_size = 45, italic = True, background_color = [1, 50, 50, 1])

		button_1.bind(on_press = self.biometric_verification)
		button_2.bind(on_press = self.show_attendance)
		button_3.bind(on_press = self.add_user)
		button_4.bind(on_press = self.remove_user)

		buttons.add_widget(button_1)
		buttons.add_widget(button_2)
		buttons.add_widget(button_3)
		buttons.add_widget(button_4)

		self.add_widget(buttons)


	def biometric_verification(self, instance):
		global usernames, embeddings
		embeddings, usernames = readAllBlobData()
		UI_interface.screen_manager.current = "Authentication"

	def show_attendance(self, instance):
		global usernames, embeddings
		global show_attendance_obj
		embeddings, usernames = readAllBlobData()

		show_attendance_obj.show()
		UI_interface.screen_manager.current = "Show_Attendance"

	def add_user(self, instance):
		UI_interface.screen_manager.current = "Add_User"

	def remove_user(self, instance):
		global remove_user_obj
		remove_user_obj.show()
		UI_interface.screen_manager.current = "Remove_User"


class Authentication_Page(GridLayout):
	
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.output_path = 'Output/'
		self.flag = 0
		self.cols = 2
		self.padding = [100, 100, 100, 100]
		self.spacing = [20, 20]
		self.begin = Button(text='BEGIN', font_size = 30, italic = True, background_color = [1, 255, 1, 1])
		self.begin.bind(on_press = self.start)
		self.add_widget(self.begin)
		self.back = Button(text='GO BACK', font_size = 30, italic = True, background_color = [255, 1, 1, 1])
		self.back.bind(on_press = self.goback)
		self.add_widget(self.back)


	def start(self, instance):
		global camera_index
		self.flag = 1
		self.img=Image()
		self.layout = BoxLayout()
		self.layout.add_widget(self.img)
		self.remove_widget(self.begin)
		self.remove_widget(self.back)
		self.add_widget(self.layout)

		mark = Button(text='VERIFY ME', font_size = 30, italic = True, background_color = [255, 1, 1, 1])
		back = Button(text='GO BACK', font_size = 30, italic = True, background_color = [1, 1, 255, 1])
		self.label = Label(text='Authenticate Yourself!!', font_size = 38, color = [255, 255, 255, 1])

		mark.bind(on_press = self.recognize)
		back.bind(on_press = self.goback)

		self.button_layout = GridLayout(rows = 3, spacing = [20, 20])

		self.button_layout.add_widget(mark)
		self.button_layout.add_widget(back)
		self.button_layout.add_widget(self.label)
		
		self.add_widget(self.button_layout)

		self.capture = cv2.VideoCapture(camera_index)
		self.event = Clock.schedule_interval(self.update, 1.0/33.0)


	def update(self, instance):
		_, self.frame = self.capture.read()
		self.frame = extract_all_faces(self.frame)
		buf1 = cv2.flip(self.frame, 0)
		buf = buf1.tostring()
		texture = Texture.create(size=(self.frame.shape[1], self.frame.shape[0]), colorfmt='bgr')
		texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
		self.img.texture = texture


	def recognize(self, instance):
		ts = datetime.datetime.now()
		img_name = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
		img_path = self.output_path + img_name
		cv2.imwrite(img_path, self.frame)
		print("[INFO] saved {}".format(img_name))
		
		if image_enhance == 'on':
			if image_enhance_use_gpu == 'on':
				os.system('sh image_enhance_gpu.sh')
			elif image_enhance_use_gpu == 'off':
				os.system('sh image_enhance_cpu.sh')

		embedding, flag = generate_embedding(img_path)
		os.system('rm ./Output/*')
		if embeddings is not None:
			if flag == 1:
				ones_matrix = np.ones((len(usernames), 1))
				embedding_matrix = np.matmul(ones_matrix, embedding.detach().numpy())
				distances = calc_distance(embedding_matrix, embeddings)
				if (distances[np.argmin(distances)] < 1.0000):
					print(usernames[np.argmin(distances)] + ' Marked')
					self.button_layout.remove_widget(self.label)
					self.label = Label(text=usernames[np.argmin(distances)] + ' Marked', font_size = 38, color = [255, 255, 255, 1])
					self.button_layout.add_widget(self.label)
					attendance[usernames[np.argmin(distances)]] = "Present"
				else:
					self.button_layout.remove_widget(self.label)
					self.label = Label(text = 'User Not Registered', font_size = 38, color = [255, 255, 255, 1])
					self.button_layout.add_widget(self.label)
			else:
				self.button_layout.remove_widget(self.label)
				self.label = Label(text='Zero/Muliple Faces Detected', font_size = 38, color = [255, 255, 255, 1])
				self.button_layout.add_widget(self.label)
		else:
			self.button_layout.remove_widget(self.label)
			self.label = Label(text='No Registered Users', font_size = 38, color = [255, 255, 255, 1])
			self.button_layout.add_widget(self.label)


	def goback(self, instance):

		if self.flag == 1:
			self.event.cancel()
			self.capture.release()
			self.remove_widget(self.layout)
			self.remove_widget(self.button_layout)
			self.__init__()
		UI_interface.screen_manager.current = "Home"


class Add_User_Page(GridLayout):
	
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.output_path = 'Output/'
		self.cols = 2
		self.flag = 0
		self.padding = [100, 100, 100, 100]
		self.spacing = [20, 20]
		self.begin = Button(text='BEGIN', font_size = 30, italic = True, background_color = [1, 255, 1, 1])
		self.begin.bind(on_press = self.start)
		self.add_widget(self.begin)
		self.back = Button(text='GO BACK', font_size = 30, italic = True, background_color = [255, 1, 1, 1])
		self.back.bind(on_press = self.goback)
		self.add_widget(self.back)

	def start(self, instance):
		global camera_index
		self.flag = 1
		self.img=Image()
		self.layout = BoxLayout()
		self.layout.add_widget(self.img)
		self.remove_widget(self.begin)
		self.remove_widget(self.back)
		self.add_widget(self.layout)

		self.name = TextInput(multiline = False, size_hint = (.2, None), height = 40)
		add = Button(text='ADD ME', font_size = 30, italic = True, background_color = [255, 1, 1, 1])
		back = Button(text='GO BACK', font_size = 30, italic = True, background_color = [1, 1, 255, 1])
		self.label = Label(text='Add Yourself!!', font_size = 38, color = [255, 255, 255, 1])


		add.bind(on_press = self.add)
		back.bind(on_press = self.goback)

		self.button_layout = GridLayout(rows = 4, spacing = [20, 20])

		self.button_layout.add_widget(self.name)
		self.button_layout.add_widget(add)
		self.button_layout.add_widget(back)
		self.button_layout.add_widget(self.label)
		
		self.add_widget(self.button_layout)

		self.capture = cv2.VideoCapture(camera_index)
		self.event = Clock.schedule_interval(self.update, 1.0/33.0)


	def update(self, instance):
		_, self.frame = self.capture.read()
		self.frame = extract_all_faces(self.frame)
		buf1 = cv2.flip(self.frame, 0)
		buf = buf1.tostring()
		texture = Texture.create(size=(self.frame.shape[1], self.frame.shape[0]), colorfmt='bgr')
		texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
		self.img.texture = texture


	def add(self, instance):
		if len(self.name.text) != 0:
			ts = datetime.datetime.now()
			img_name = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
			img_path = self.output_path + img_name
			cv2.imwrite(img_path, self.frame)
			print("[INFO] saved {}".format(img_name))

			if image_enhance == 'on':
				if image_enhance_use_gpu == 'on':
					os.system('sh image_enhance_gpu.sh')
				elif image_enhance_use_gpu == 'off':
					os.system('sh image_enhance_cpu.sh')

			embedding, flag = generate_embedding(img_path)
			os.system('rm ./Output/*')
			if flag == 1:
				insertBLOB(self.name.text, embedding)
				self.button_layout.remove_widget(self.label)
				self.label = Label(text=self.name.text + ' Added', font_size = 38, color = [255, 255, 255, 1])
				self.button_layout.add_widget(self.label)
			else:
				self.button_layout.remove_widget(self.label)
				self.label = Label(text = 'Zero/Multiple Faces Detected', font_size = 38, color = [255, 255, 255, 1])
				self.button_layout.add_widget(self.label)
		else:
			self.button_layout.remove_widget(self.label)
			self.label = Label(text = 'Enter UserName', font_size = 38, color = [255, 255, 255, 1])
			self.button_layout.add_widget(self.label)

	def goback(self, instance):
		global usernames, embeddings

		if self.flag == 1:
			self.event.cancel()
			self.capture.release()
			self.remove_widget(self.layout)
			self.remove_widget(self.button_layout)
			self.__init__()

			embeddings, usernames = readAllBlobData()
			for username in usernames:
				if username not in attendance.keys():
					attendance[username] = 'Absent' 
		print(attendance)
		UI_interface.screen_manager.current = "Home"



class Show_Attendance_Page(GridLayout):
	
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.flag = 0
		self.cols = 2
		self.padding = [100, 100, 100, 100]
		self.spacing = [20, 20]


	def show(self):
		self.attendance_list = GridLayout(cols = 2, rows = 42, spacing = [20, 20])
		for key in attendance.keys():
			self.attendance_list.add_widget(Label(text = key, font_size = 20, color = [255, 255, 255, 1]))
			self.attendance_list.add_widget(Label(text = attendance[key], font_size = 20, color = [255, 255, 255, 1]))

		self.add_widget(self.attendance_list)

		self.back = Button(text='GO BACK', font_size = 30, italic = True, background_color = [255, 1, 1, 1])
		self.back.bind(on_press = self.goback)
		self.add_widget(self.back)

	def goback(self, instance):

		self.remove_widget(self.back)
		self.remove_widget(self.attendance_list)
		UI_interface.screen_manager.current = "Home"


class Remove_User_Page(GridLayout):
	
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.flag = 0
		self.cols = 2
		self.padding = [100, 100, 100, 100]
		self.spacing = [20, 20]


	def show(self):
		self.name = TextInput(multiline = False, size_hint = (.2, None), height = 40)
		self.remove = Button(text='REMOVE USER', font_size = 30, italic = True, background_color = [255, 1, 1, 1])
		self.remove.bind(on_press = self.remove_user)
		self.back = Button(text='GO BACK', font_size = 30, italic = True, background_color = [255, 1, 1, 1])
		self.back.bind(on_press = self.goback)
		self.buttons = GridLayout(cols = 1, spacing = [20, 20])
		self.buttons.add_widget(self.name)
		self.buttons.add_widget(self.remove)
		self.buttons.add_widget(self.back)
		self.add_widget(self.buttons)

		self.attendance_list = GridLayout(cols = 2, rows = 42, spacing = [20, 20])
		for key in attendance.keys():
			self.attendance_list.add_widget(Label(text = key, font_size = 20, color = [255, 255, 255, 1]))
			self.attendance_list.add_widget(Label(text = attendance[key], font_size = 20, color = [255, 255, 255, 1]))

		self.add_widget(self.attendance_list)

	def remove_user(self, instance):
		if len(self.name.text) != 0:
			deleteBlob(self.name.text)
			self.remove_widget(self.attendance_list)
			if self.name.text in attendance:
				del attendance[self.name.text]
			print(attendance)
			self.attendance_list = GridLayout(cols = 2, rows = 42, spacing = [20, 20])
			for key in attendance.keys():
				self.attendance_list.add_widget(Label(text = key, font_size = 20, color = [255, 255, 255, 1]))
				self.attendance_list.add_widget(Label(text = attendance[key], font_size = 20, color = [255, 255, 255, 1]))

			self.add_widget(self.attendance_list)


	def goback(self, instance):
		self.remove_widget(self.buttons)
		self.remove_widget(self.attendance_list)
		UI_interface.screen_manager.current = "Home"



class RBVMS(App):

	def build(self):
		global show_attendance_obj, remove_user_obj
		self.screen_manager = ScreenManager()

		self.home_page = Home_Page()
		screen = Screen(name='Home')
		screen.add_widget(self.home_page)
		self.screen_manager.add_widget(screen)

		self.authentication_page = Authentication_Page()
		screen = Screen(name='Authentication')
		screen.add_widget(self.authentication_page)
		self.screen_manager.add_widget(screen)

		self.add_user = Add_User_Page()
		screen = Screen(name='Add_User')
		screen.add_widget(self.add_user)
		self.screen_manager.add_widget(screen)

		show_attendance_obj = Show_Attendance_Page()
		screen = Screen(name='Show_Attendance')
		screen.add_widget(show_attendance_obj)
		self.screen_manager.add_widget(screen)	

		remove_user_obj = Remove_User_Page()
		screen = Screen(name='Remove_User')
		screen.add_widget(remove_user_obj)
		self.screen_manager.add_widget(screen)	

		return self.screen_manager


if __name__ == "__main__":

	if not os.path.exists('Zero-DCE_code/data/'):
		os.makedirs('Zero-DCE_code/data/')
		os.makedirs('Zero-DCE_code/data/result/')
		os.makedirs('Zero-DCE_code/data/test_data/')
		os.makedirs('Zero-DCE_code/data/test_data/test/')
	
	if not os.path.exists('Output/'):
		os.makedirs('Output/')

	if not os.path.exists('Face_Database.db'):
		create_table()

	embeddings, usernames = readAllBlobData()
	
	for username in usernames:
		if username not in attendance.keys():
			attendance[username] = 'Absent' 
	
	UI_interface = RBVMS()
	UI_interface.run()
