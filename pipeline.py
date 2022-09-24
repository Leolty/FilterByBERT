from baseline import Semi_EM_NB, Baseline_Model
from bert import BERT_Model
from configure import APP_NAME
from sklearn import naive_bayes, neural_network, linear_model, svm, ensemble

def main():
    print("Training NB...")
    nb_acc, nb_f = Baseline_Model(APP_NAME, naive_bayes.MultinomialNB())
    
    print("Training LR...")
    lr_acc, lr_f = Baseline_Model(APP_NAME, linear_model.LogisticRegression())

    print("Training RF")
    rf_acc, rf_f = Baseline_Model(APP_NAME, ensemble.RandomForestClassifier())
    
    print("Training SVM...")
    svm_acc, svm_f = Baseline_Model(APP_NAME, svm.SVC())
    
    print("Training MLP (512)...")
    mlp512_acc, mlp512_f = Baseline_Model(APP_NAME, neural_network.MLPClassifier(hidden_layer_sizes=(512,)))
    
    print("Training MLP (512,256)...")
    mlp512_256_acc, mlp512_256_f = Baseline_Model(APP_NAME, neural_network.MLPClassifier(hidden_layer_sizes=(512,256,)))

    print("Training Semi_EMNB..")
    emnb_acc, emnb_f =Semi_EM_NB(APP_NAME)

    print("fine Tune BERT-base-0")
    bert_base_0_acc, bert_base_0_f = BERT_Model(APP_NAME, "bert-base-uncased", 12, 16, 25)
    
    print("Fine Tune BERT-base-6")
    bert_base_6_acc, bert_base_6_f = BERT_Model(APP_NAME, "bert-base-uncased", 6, 16, 20)

    print("Fine Tune BERT-base-10")
    bert_base_10_acc, bert_base_10_f = BERT_Model(APP_NAME, "bert-base-uncased", 2, 16, 15)

    print("fine Tune BERT-large-0")
    bert_large_0_acc, bert_large_0_f = BERT_Model(APP_NAME, "bert-large-uncased", 12, 16, 25)
    
    print("Fine Tune BERT-large-6")
    bert_large_6_acc, bert_large_6_f = BERT_Model(APP_NAME, "bert-large-uncased", 6, 16, 20)

    print("Fine Tune BERT-large-10")
    bert_large_10_acc, bert_large_10_f = BERT_Model(APP_NAME, "bert-large-uncased", 2, 16, 15)

    print("NB", nb_f)
    print("LR", lr_f)
    print("RF", rf_f)
    print("SVM", svm_f)
    print("MLP-512", mlp512_f)
    print("MLP-(512,256)", mlp512_256_f)
    print("SEMI-EMNB", emnb_f)
    print("BERT-BASE-0", bert_base_0_f)
    print("BERT-BASE-6", bert_base_6_f)
    print("BERT-BASE-10", bert_base_10_f)
    print("BERT-LARGE-0", bert_large_0_f)
    print("BERT-LARGE-6", bert_large_6_f)
    print("BERT-LARGE-10", bert_large_10_f)
    
if __name__ == "__main__":
    main()