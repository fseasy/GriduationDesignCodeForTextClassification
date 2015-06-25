#!/usr/bin/env python
#coding=utf-8

try :
    import cPickle as pickle
except :
    import pickle 
import sys
import os
import logging
import traceback
import math

sys.path.append("../seg/")

import build_stop_words
build_stop_words.load_stop_words("../seg/stop_words")

POSITIVE_TYPE = "t1"


class word_bag :
    docs = {}
    features = {}
    dtypes = {}
    CHI = {}
    N = 0
    top_k_feature = {}
    def add_word_bag(self,dtype , doc , feature) :
        # build dtypes
        if dtype not in self.dtypes :
            self.dtypes[dtype] = {doc:True} # for fast
        else :
            if doc not in self.dtypes[dtype] :
                self.dtypes[dtype][doc] = True
        #build feature
        if feature not in self.features :
            self.features[feature] = {
                        "docs" : [doc ,] ,
                        "dtypes" : {dtype : 1}
                    }
        else :
            if doc not in self.features[feature]["docs"] :
                self.features[feature]["docs"].append(doc)
                if dtype not in self.features[feature]["dtypes"] :
                    self.features[feature]["dtypes"][dtype] = 1 
                else :
                    self.features[feature]["dtypes"][dtype] += 1
        #bulid doc
        if doc not in self.docs :
            self.docs[doc] = {
                        "dtype" : dtype ,
                        "feature_info" : { feature : 1}
                    }
        else :
            if feature not in self.docs[doc]["feature_info"] :
                self.docs[doc]["feature_info"][feature] = 1 
            else :
                self.docs[doc]["feature_info"][feature] += 1
    def save_word_bag(self) :
        out_file = open("word_bag.pickle" , 'w')
        out_obj = {
                "docs" : self.docs ,
                "features" : self.features ,
                "dtypes" : self.dtypes
                }

        pickle.dump(out_obj , out_file)
        out_file.close()

    def load_word_bag(self) :
        try :
            read_file = open("word_bag.pickle")
            read_obj = pickle.load(read_file)
            self.docs = read_obj["docs"]
            self.features = read_obj["features"]
            self.dtypes = read_obj["dtypes"]
            load_state = True
            read_file.close()
        except Exception , e:
            print >> sys.stderr , "failed to load '%s'" %("pickle file")
            print >> sys.stderr , e
            load_state = False
        finally :
            return load_state
    
    def dump_to_file(self) :
        doc_file = open("doc.txt" , "w")
        for key in self.docs :
            doc_file.write("doc  %s : \n" %(key))
            doc_file.write("\ttype : %s\n\tfeatures:\n" %(self.docs[key]["dtype"]))
            for f in self.docs[key]["feature_info"] :
                doc_file.write("\t\t%s : %d\n" %(f.encode("utf-8") , self.docs[key]["feature_info"][f]))
        doc_file.close()

        feature_file = open("feature.txt" , "w")
        for key in self.features :
            feature_file.write("feature : %s\n" %(key.encode("utf-8")))
            for d in self.features[key]["docs"] :
                feature_file.write("\t%s\n" %(d))
            for t in self.features[key]["dtypes"] :
                feature_file.write("\t%s:%d\n"  %(t , self.features[key]["dtypes"][t]))
        feature_file.close()

        type_file = open("type.txt" , "w")
        for key in self.dtypes :
            type_file.write("type : %s\n"  %(key))
            for d in self.dtypes[key] :
                type_file.write("\t%s\n" %(d))
        type_file.close()
    
    def dump_feature_DF(self) :
        f_df = []
        for f in self.features :
            f_df.append((f , len(self.features[f]["docs"])))
        f_df = sorted(f_df , key=lambda x : x[1] , reverse=True)
        f_df_f = open("feature_DF.txt" , "w")
        for item in f_df :
            f_df_f.write("%s\t%d\n"  %(item[0].encode("utf-8") , item[1]))
        f_df_f.close()

    def filter_feature_by_DF(self , df_threshold) :
        feature_abandon = []
        for f in self.features :
            df = len(self.features[f]["docs"])
            if df <= df_threshold :
                doc_list = self.features[f]["docs"]
                for d in doc_list :
                    self.docs[d]["feature_info"].pop(f) # remove the feature in the docs
                # self.features.pop(f) # can not do it here
                feature_abandon.append(f)
        for f in feature_abandon :
            self.features.pop(f)
    
    def calc_all_feature_with_type_CHI(self , dtype) :
        self.CHI[dtype] = {}
        self.N = len(self.docs)
        for f in self.features :
            # calc A : the DF which :  belongs to dtype and contains this feature   
            A = 0 
            if dtype in self.features[f]["dtypes"] :
                A = self.features[f]["dtypes"][dtype]
            # calc B : the DF which : not belongs to dtype but contains this feature
            ##for t in self.features[f]["dtypes"] :
            ##    if t != dtype :
            ##        B += self.features[f]["dtypes"][t]
            B = len(self.features[f]["docs"]) - A
            #calc C : the DF : not belongs to dtype but not contains this feature
            C = 0
            belong_docs = self.dtypes[dtype]
            for d in belong_docs :
                if f not in self.docs[d]["feature_info"] :
                    C += 1
            #calc D : the DF : not belogs to dtype and not contains this feature 
            ##for d in self.docs :
            ##    if ( f not in self.docs[d]["feature_info"] ) and ( self.docs[d]["dtype"] != dtype ) :
            ##        D += 1
            D = len(self.docs) - A -B -C 
            #print "[A = %d , B = %d , C = %d , D = %d ] TOTAL = %d " %(A , B , C , D , A+B+C+D)
            chi = (self.N*(A*D - C*B)**2) / float((A+C)*(B+D)*(A+B)*(C+D))
            self.CHI[dtype][f] = chi
    def dump_top_k_feature_with_type_by_CHI(self , dtype , k) :
        sorted_rst = sorted(self.CHI[dtype].items() , key=lambda x : x[1] , reverse=True)
        out_file = open("top%d.txt" %(k) , "w")
        for x in range(k) :
            if x >= len(sorted_rst) :
                break
            out_file.write("%d\t%s\t%f\t%d\n"  %(x+1 , sorted_rst[x][0].encode("utf-8") , sorted_rst[x][1] , len(self.features[sorted_rst[x][0]]["docs"])))
        out_file.close()
    
    def get_top_k_feature_with_type_by_CHI(self , dtype , k) :
        if dtype not in self.CHI :
            return
        sorted_rst = sorted(self.CHI[dtype].items() , key=lambda x : x[1] ,reverse=True)
        self.top_k_feature[dtype] = [x[0] for x in sorted_rst[:k]]

    def write_feature_with_type(self , dtype) :
        if dtype not in self.top_k_feature :
            return
        out_file = open("feature_info.txt" , "w")
        idx = 1 
        for f in self.top_k_feature[dtype] :
            idf = math.log(self.N / len(self.features[f]["docs"]))
            out_file.write("%s\t%d\t%f\n" %(f.encode("utf-8") , idx , idf))
            idx += 1 
        out_file.close()

def is_valid(word) :
    if word not in build_stop_words.stop_words :
        return True
    else :
        return False


def build_word_bag( training_set_dir , word_bag) :
    if not os.path.exists(training_set_dir) :
       logging.error("training set dir '%s' does not exists" %training_set_dir)
       return
    
    file_list = os.listdir(training_set_dir) 

    for file_name in file_list :
        dtype = file_name.split('_')[0]
        doc = os.path.splitext(file_name)[0]
        path = os.path.join(training_set_dir , file_name)
        try :
            read_file = open(path)
            for line in read_file :
                try :
                    line = line.strip()
                    line = line.decode("utf-8")
                    words = line.split()
                    for word in words :
                        if is_valid(word) :
                            word_bag.add_word_bag(dtype , doc , word )
                except Exception , e :
                    logging.warning("line decode error : %s" %(e))
        except Exception , e :
            logging.error("failed to open file '%s'" %(path))
        finally :
            read_file.close()




if __name__ == "__main__" :
    if len(sys.argv) != 2 :
        logging.error("usage : %s [%s]"  %(sys.argv[0] , "target_dir"))
        exit(0)
    wb = word_bag() ;
    load_state = wb.load_word_bag()
    if not load_state :
        build_word_bag(sys.argv[1] , wb)
        wb.save_word_bag()
    #wb.dump_to_file() ;
    #wb.dump_feature_DF()
    #wb.filter_feature_by_DF(6)
    #wb.dump_feature_DF()
    #wb.save_word_bag() 
    wb.calc_all_feature_with_type_CHI(POSITIVE_TYPE)
    wb.get_top_k_feature_with_type_by_CHI(POSITIVE_TYPE , 5000)
    wb.write_feature_with_type(POSITIVE_TYPE)
