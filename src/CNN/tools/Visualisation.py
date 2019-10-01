import numpy as np
import matplotlib.pyplot as plt
import mdtraj as md
import pandas as pd


class Visualisation:

    def fastVoxelPlot(self, data, complexid, featureid):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.voxels(data.dataset[complexid, :, :, :, featureid])
        ax.voxels(data.dataset[complexid, :, :, :, featureid + 8])
        plt.show()

    def visComplex(self, data, featureid):
        self.create_pdb(data, data.feature[:, featureid], 'prot.pdb')
        self.create_pdb(data, data.feature2[:, featureid], 'lig.pdb')
        # TODO: Here VMD and NGLView will be implemented to show up the Result with correct Atom-selection
        '''
        struc = nv.FileStructure('1atp.pdb')
        view = nv.NGLWidget(struc)
        view
        '''

    def create_pdb(self, data, val, pdb_name):
        serial = list(range(0, 13824))
        name = ['test'] * 13824
        element = np.array(['C'] * 13824)
        element[val > 0] = 'O'
        resSeq = list(np.zeros(13824))
        resName = ['test'] * 13824
        chainID = list(np.zeros(13824))
        topo = {
            'serial': serial,
            'name': name,
            'element': element,
            'resSeq': resSeq,
            'resName': resName,
            'chainID': chainID
        }

        topology = md.Topology.from_dataframe(pd.DataFrame(topo))
        pdb = md.formats.PDBTrajectoryFile(pdb_name, mode='w')
        pdb.write(data.center, topology)
