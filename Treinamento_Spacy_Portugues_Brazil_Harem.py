#!/usr/bin/env python
# coding: utf8
from __future__ import unicode_literals, print_function
#
import plac
import random
from pathlib import Path
import spacy
import pandas as pd
import codecs
from lxml import etree
from selenium import webdriver
from spacy.gold import GoldParse
from spacy.scorer import Scorer


#Carrega os dados do Corpus Harem
tree = etree.parse('./harem/lampada2.0/coleccoes/CDSegundoHAREM_TEMPO.xml')
texto = etree.parse('./harem/lampada2.0/coleccoes/colSegundoHAREM.xml')

#Extrai apenas o texto
texto = [i.text for i in texto.findall('.//*')]
texto = [i.replace('\n','') for i in texto]

#Extrai as entidades do tipo 'Pessoa' para treinamento
nomes = [i.text for i in tree.xpath('.//EM[@CATEG="PESSOA" and @TIPO="INDIVIDUAL"]')]

#Elimina repetições
setnomes = set(nomes)

#Divida a lista de textos em 2 para que possamos treinar e validar
n = 2
splited = [texto[i::n] for i in range(n)]
#print(splited[1]) 

texto = splited[0]
texto_teste = splited[1]

#Identifica as entidades dentro do texto de treinamento e localiza a posição de inicio e fim da entidade e adiciona a lista de treino, 
TRAIN_DATA = []
TESTE_DATA = []

for i in texto:

    dict_text = {'texto': i,
                 'entidades': []}
    for j in setnomes:
        if j in i:
            start = i.index(j)
            end = start + len(i)
            dict_text['entidades'].append((start, end, 'PESSOA'))

    if len(dict_text['entidades']):
        temp = (dict_text['texto'], {'entities': dict_text['entidades']})
        TRAIN_DATA.append(temp)



#Identifica as entidades para validação e adiciona na lista de teste
for i in texto_teste:

    dict_text = {'texto': i,
                 'entidades': []}
    for j in setnomes:
        if j in i:
            start = i.index(j)
            end = start + len(i)
            dict_text['entidades'].append((start, end, 'PESSOA'))

    if len(dict_text['entidades']):
        temp = (dict_text['texto'], dict_text['entidades'])
        TESTE_DATA.append(temp)

#Esta função avalia os dados de teste com a saida das métricas para mais informações acessar a documentação oficial (spacy)
def evaluate(ner_model, examples):
    scorer = Scorer()
    for input_, annot in examples:
        doc_gold_text = ner_model.make_doc(input_)
        gold = GoldParse(doc_gold_text, entities=annot)
        pred_value = ner_model(input_)
        scorer.score(pred_value, gold)
    return scorer.scores



#Criação do modelo (código disponivel na pagina oficial do spacy)
@plac.annotations(
    model=('.\model'),
    output_dir=('.\model'),
    n_iter=("Number of training iterations", "option", "n", int))


def main(model=None, output_dir=None, n_iter=100):

    """Load the model, set up the pipeline and train the entity recognizer."""
    output_dir = './modelo' #coloquei para que fosse possivel salvar o modelo######
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('pt')  # create blank Language class
        print("Created blank 'pt' model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe('ner')

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print("LOSS:",losses)

    # Testa o modelo Treinado
    for text, _ in TRAIN_DATA:
        doc = nlp(text)
        print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
        print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])

    # Salva o modelo no diretório
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        for text, _ in TRAIN_DATA:
            doc = nlp2(text)
            print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
            print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])


if __name__ == '__main__':
    plac.call(main)


#Executa a validação
ner_model = spacy.load('./modelo')
results = evaluate(ner_model, TESTE_DATA)
print(results)
