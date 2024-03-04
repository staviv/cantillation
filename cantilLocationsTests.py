import unittest
from  cantilLocations import cantilLocationsObject
import nikud_and_teamim




class TestcantilLocations(unittest.TestCase):

    def test_1(self):
        cantilLocationsObject1 = cantilLocationsObject("בעמ֤וד ענן֙ לנחת֣ם הד֔רך")
        self.assertEqual(cantilLocationsObject1.get_list(), [(19, 0, 2), (8, 1, 2), (18, 2, 3), (3, 3, 1)])


    def test_2(self):
        cantilLocationsObject1 = cantilLocationsObject("וַיֹּ֤אמֶר אֵלָיו֙ הָעֶ֔בֶד")
        self.assertEqual(cantilLocationsObject1.get_list(), [(19, 0, 1), (8, 1, 3), (3, 2, 1)])


    def test_2(self):
        cantilLocationsObject1 = cantilLocationsObject("וַיֹּ֤אמֶר אֵלָיו֙ הָעֶ֔בֶד אוּלַי֙ לֹא־תֹאבֶ֣ה הָֽאִשָּׁ֔ה לָלֶ֥כֶת אַחֲרַ֖י אֶל־הָאָ֣רֶץ הַזֹּ֑את")
        self.assertEqual(cantilLocationsObject1.get_list(), [(19, 0, 1), (8, 1, 3), (3, 2, 1) , (8, 3, 3) ,(18, 5, 2),(30, 6, 0) ,(3, 6, 2) ,(20, 7, 1), (5, 8, 2), (18, 10, 1), (0, 11, 1)  ])

    def test_3(self):
        cantilLocationsObject1 = cantilLocationsObject("וְאֵ֗לֶּה שְׁנֵי֙ חַיֵּ֣י יִשְׁמָעֵ֔אל")
        self.assertEqual(cantilLocationsObject1.get_list(), [(6, 0, 1), (8, 1, 2), (18, 2, 1), (3, 3, 3)])


    def test_4(self):
        cantilLocationsObject1 = cantilLocationsObject("וְהָיְתָ֥ה הַקֶּ֖שֶׁת בֶּֽעָנָ֑ן וּרְאִיתִ֗יהָ לִזְכֹּר֙ בְּרִ֣ית עוֹלָ֔ם")
        self.assertEqual(cantilLocationsObject1.get_list(), [(20, 0, 3), (5, 1, 1), (30, 2, 0), (0, 2, 2), (6, 3, 4), (8, 4, 3), (18, 5, 1), (3, 6, 2)])


    def test_5(self):
        cantilLocationsObject1 = cantilLocationsObject("וַתַּשְׁקֶ֜יןָ גַּ֣ם בַּלַּ֧יְלָה הַה֛וּא אֶת־אֲבִיהֶ֖ן יָ֑יִן וַתָּ֤קׇם הַצְּעִירָה֙ וַתִּשְׁכַּ֣ב עִמּ֔וֹ וְלֹֽא־יָדַ֥ע בְּשִׁכְבָ֖הּ וּבְקֻמָֽהּ")
        self.assertEqual(cantilLocationsObject1.get_list(), [(11, 0, 3), (18, 1, 0), (22, 2, 1), (10, 3, 1), (5, 5, 3), (0, 6, 0), (19, 7, 1), (8, 8, 5), (18, 9, 3), (3, 10, 1), (30, 11, 1), (20, 12, 1), (5, 13, 3), (30, 14, 3)])        


if __name__ == '__main__':
    unittest.main()