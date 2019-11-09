import sys
sys.path.append('/home/max/Dokumente/Masterarbeit/PredBind')
from src.SchNet.tools.CreateFeatureset import CreateFeatureset

CreateFeatureset.createFeatureset(datapath='../../../Data/test/',
                                  indexpath='../../../Data/INDEX_refined_data.2016.2018',
                                  targetpath='../../../Data//temp/new/')
