import unittest

from ..cantilLocations import cantilLocationsObject
from .. import nikud_and_teamim
from ..cantilLocations_evaluation import calculate_TP_FP_FN_Exact,calculate_TP_FP_FN_with_Letter_Shift,calculate_TP_FP_FN_Word_Level,calculate_TP_FP_FN_Word_Level_with_Word_Shift
from ..cantilLocations_evaluation import calculate_TP_FP_FN,calculate_precision_recall_f1,calculate_precision_recall_f1_for_string,calculate_precision_recall_f1_for_string_list


class Test_cantilLocations_evaluation(unittest.TestCase):

    def test_calculate_TP_FP_FN_Exact(self):
        tuple1 = calculate_TP_FP_FN_Exact(cantilLocationsObject("בעמ֤וד ענן֙ לנחת֣ם הד֔רך") ,cantilLocationsObject("בעמ֤וד ענן֙ לנחת֣ם הד֔רך") ) 
        self.assertEqual(tuple1, (4, 0, 0))
        tuple2 = calculate_TP_FP_FN_Exact(cantilLocationsObject("בעמ֤וד ענן֙ לנחת֣ם הד֔רך") ,cantilLocationsObject("בעמוד ענן֙ לנחת֣ם הד֔רך") ) 
        self.assertEqual(tuple2, (3, 0, 1))
        tuple3 = calculate_TP_FP_FN_Exact(cantilLocationsObject("בעמ֤וד ענן֙ לנחת֣ם הד֔רך") ,cantilLocationsObject("בעמ֤וד ענן֙ לנחֹ֣ת֣ם הד֔רך") ) 
        self.assertEqual(tuple3, (4, 1, 0))

    def test_calculate_TP_FP_FN_with_Letter_Shift(self):
        #should be the same as Exact
        tuple1 = calculate_TP_FP_FN_with_Letter_Shift(cantilLocationsObject("בעמ֤וד ענן֙ לנחת֣ם הד֔רך") ,cantilLocationsObject("בעמ֤וד ענן֙ לנחת֣ם הד֔רך") ) 
        self.assertEqual(tuple1, (4, 0, 0))
        #shift -  בע֤מוד  instead of בעמ֤וד
        tuple2 = calculate_TP_FP_FN_with_Letter_Shift(cantilLocationsObject("בעמ֤וד ענן֙ לנחת֣ם הד֔רך") ,cantilLocationsObject("בע֤מוד ענן֙ לנחת֣ם הד֔רך") ) 
        self.assertEqual(tuple2, (4, 0, 0))

    def test_calculate_TP_FP_FN_Word_Level(self):
        #the same str , the same as Exact
        tuple1 = calculate_TP_FP_FN_Word_Level(cantilLocationsObject("בעמ֤וד ענן֙ לנחת֣ם הד֔רך") ,cantilLocationsObject("בעמ֤וד ענן֙ לנחת֣ם הד֔רך") ) 
        self.assertEqual(tuple1, (4, 0, 0))
        #shift -  בע֤מוד  instead of בעמ֤וד
        tuple2 = calculate_TP_FP_FN_Word_Level(cantilLocationsObject("בעמ֤וד ענן֙ לנחת֣ם הד֔רך") ,cantilLocationsObject("בע֤מוד ענן֙ לנחת֣ם הד֔רך") ) 
        self.assertEqual(tuple2, (4, 0, 0))
        #shift -  בעמוד  instead of בעמ֤וד
        tuple3 = calculate_TP_FP_FN_Word_Level(cantilLocationsObject("בעמ֤וד ענן֙ לנחת֣ם הד֔רך") ,cantilLocationsObject("בעמוד ענן֙ לנחת֣ם הד֔רך") ) 
        self.assertEqual(tuple3, (3, 0, 1))

    def test_calculate_TP_FP_FN_Word_Level_with_Word_Shift(self):
        #the same str , the same as Exact
        tuple1 = calculate_TP_FP_FN_Word_Level_with_Word_Shift(cantilLocationsObject("בעמ֤וד ענן֙ לנחת֣ם הד֔רך") ,cantilLocationsObject("בעמ֤וד ענן֙ לנחת֣ם הד֔רך") ) 
        self.assertEqual(tuple1, (4, 0, 0))
        #  ע֤ in ענן  - shift to the next word
        tuple2 = calculate_TP_FP_FN_Word_Level_with_Word_Shift(cantilLocationsObject("בעמ֤וד ענן֙ לנחת֣ם הד֔רך") ,cantilLocationsObject("בעמוד ע֤נן֙ לנחת֣ם הד֔רך") ) 
        self.assertEqual(tuple2, (4, 0, 0))
        #shift -  בעמוד  instead of בעמ֤וד
        tuple3 = calculate_TP_FP_FN_Word_Level_with_Word_Shift(cantilLocationsObject("בעמ֤וד ענן֙ לנחת֣ם הד֔רך") ,cantilLocationsObject("בעמוד ענן֙ לנחת֣ם הד֔רך") ) 
        self.assertEqual(tuple3, (3, 0, 1))

    def test_calculate_precision_recall_f1(self):
        tuple1 = calculate_precision_recall_f1(cantilLocationsObject("בעמ֤וד ענן֙ לנחת֣ם הד֔רך") ,cantilLocationsObject("בעמ֤וד ענן֙ לנחת֣ם הד֔רך") ) 
        self.assertEqual(tuple1, (1, 1, 1))
        tuple2 = calculate_precision_recall_f1(cantilLocationsObject("בעמ֤וד ענן֙ לנחת֣ם הד֔רך") ,cantilLocationsObject("בעמוד ענן֙ לנחת֣ם הד֔רך") ) 
        self.assertEqual(tuple2, (1, 3/4, 0.8571428571428571))
        tuple3 = calculate_precision_recall_f1(cantilLocationsObject("בעמ֤וד ענן֙ לנחת֣ם הד֔רך") ,cantilLocationsObject("בעמ֤וד ענן֙ לנחֹ֣ת֣ם הד֔רך") ) 
        self.assertEqual(tuple3, ( 4/5, 1, 2*(4/5)/(1+4/5)))

    def test_calculate_precision_recall_f1_for_string(self):
        tuple1 = calculate_precision_recall_f1_for_string("בעמ֤וד ענן֙ לנחת֣ם הד֔רך" ,"בעמ֤וד ענן֙ לנחת֣ם הד֔רך" ) 
        self.assertEqual(tuple1, (1, 1, 1))
        tuple2 = calculate_precision_recall_f1_for_string("בעמ֤וד ענן֙ לנחת֣ם הד֔רך" ,"בעמוד ענן֙ לנחת֣ם הד֔רך" ) 
        self.assertEqual(tuple2, (1, 3/4, 0.8571428571428571))
        tuple3 = calculate_precision_recall_f1_for_string("בעמ֤וד ענן֙ לנחת֣ם הד֔רך" ,"בעמ֤וד ענן֙ לנחֹ֣ת֣ם הד֔רך" ) 
        self.assertEqual(tuple3, ( 4/5, 1, 2*(4/5)/(1+4/5)))

    def test_calculate_precision_recall_f1_for_string_list(self):  
        list1 = ["בעמ֤וד ענן֙ לנחת֣ם הד֔רך", "בעמ֤וד ענן֙ לנחת֣ם הד֔רך", "בעמ֤וד ענן֙ לנחת֣ם הד֔רך"]
        list2 = ["בעמ֤וד ענן֙ לנחת֣ם הד֔רך", "בעמוד ענן֙ לנחת֣ם הד֔רך", "בעמ֤וד ענן֙ לנחֹ֣ת֣ם הד֔רך"]
        tuple1 = calculate_precision_recall_f1_for_string_list(list1, list2)
        self.assertEqual( tuple1, ([1.0, 1.0, 0.8], [1.0, 0.75, 1.0], [1.0, 0.8571428571428571,  2*(4/5)/(1+4/5)]) )         

if __name__ == '__main__':
    unittest.main()