from selenium import webdriver
import unittest

class NewVisitorTest(unittest.TestCase):

	def setUp(self):
		self.browser = webdriver.Firefox()

	def tearDown(self):
		self.browser.quit()

	def test_can_enter_a_food_and_get_similar_ones(self):
		# Daniel has heard of a website where one can get intelligent cat food recommendations. He goes to check out the homepage.
		self.browser.get('http://localhost:5000')

		# He notices the page title and is invited to enter a food that Lenny likes.
		self.assertIn('HappyKitty',self.browser.title)
		self.fail('Finish the test!')


if __name__ == '__main__':
	unittest.main(warnings='ignore')