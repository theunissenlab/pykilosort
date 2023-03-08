import numpy as np

from ..utils import Bunch


""" Some standard probe geometries - please check before using! """

def np1_probe():
    """ Returns a Neuropixels 1 probe as a Bunch object for use in pykilosort """
    probe = Bunch()
    probe.NchanTOT = 385
    probe.chanMap = np.arange(384)
    probe.xc = np.tile(np.array([43., 11., 59., 27.]), 96)
    probe.yc = np.repeat(np.arange(20, 3841, 20.), 2)
    probe.kcoords = np.zeros(384)
    return probe


def np2_probe():
    """ Returns a Neuropixels 2 probe as a Bunch object for use in pykilosort """
    probe = Bunch()
    probe.NchanTOT = 385
    probe.chanMap = np.arange(384)
    probe.xc = np.tile(np.array([0., 32.]), 192)
    probe.yc = np.repeat(np.arange(0, 2866, 15.), 2)
    probe.kcoords = np.zeros(384)
    return probe


def get_4shank_channels_np2(shank):
    """
    Returns the channel indices for a given shank on a np2 4 shank probe
    :param shank: Shank to return, int between 0 and 3 inclusive
    :return: Numpy array
    """
    assert type(shank) == int, 'Shank index must be an integer'
    assert 0 <= shank <= 3, 'Shank index must be between 0 and 3'

    if shank in [0, 1]:
        return np.concatenate((np.arange(48*shank, 48*(shank+1)),
                                        np.arange(48*(shank+2), 48*(shank+3))))
    if shank in [2, 3]:
        return np.concatenate((np.arange(48*(shank+2), 48*(shank+3)),
                                        np.arange(48*(shank+4), 48*(shank+5))))

def np2_4shank_probe(shank=None):
    """
    Returns a Neuropixels 2 4-shank probe as a Bunch object
    :param shank: Optional, return only a single shank, int between 0 and 3 inclusive
    :return: Bunch object for pykilosort
    """
    if shank is not None:
        assert type(shank) == int, 'Shank index must be an integer'
        assert 0 <= shank <= 3, 'Shank index must be between 0 and 3'

    probe = Bunch()
    probe.NchanTOT = 385

    if shank is None:
        # Return whole probe
        probe.chanMap = np.arange(384)
        probe.kcoords = np.zeros(384)
        probe.xc = np.zeros(384)
        probe.yc = np.zeros(384)

        for shank_id in range(4):
            shank_channels = get_4shank_channels_np2(shank_id)
            probe.xc[shank_channels] = np.tile([0., 32.], 48) + shank_id * 200
            probe.yc[shank_channels] = np.repeat(np.arange(2880, 3586, 15.), 2)

        return probe

    if shank in [0, 1]:
        probe.chanMap = np.concatenate((np.arange(48*shank, 48*(shank+1)),
                                        np.arange(48*(shank+2), 48*(shank+3))))
    if shank in [2, 3]:
        probe.chanMap = np.concatenate((np.arange(48*(shank+2), 48*(shank+3)),
                                        np.arange(48*(shank+4), 48*(shank+5))))

    probe.xc = np.tile([0., 32.], 48) + shank * 200
    probe.yc = np.repeat(np.arange(2880, 3586, 15.), 2)
    probe.kcoords = np.zeros(96)

    return probe

def cambridge_neurotech_assy_236_h10():
    """ Returns a cambridge neurotech 64 channel H10 probe """
    probe = Bunch()
    probe.NchanTOT = 64
    probe.chanMap = np.arange(64)
    probe.xc = np.zeros(64)
    probe.yc = np.zeros(64)
    probe.kcoords = np.zeros(16)

    probe.chanMap = np.arange(64)#TODO 
    probe_to_intan = np.array([
                        41,39,38,37,35,34,33,32,29,30,28,26,25,24,22,20,
                        46,45,44,43,42,40,36,31,27,23,21,18,19,17,16,14,
                        55,53,54,52,51,50,49,48,47,15,13,12,11,9,10,8,
                        63,62,61,60,59,58,57,56,7,6,5,4,3,2,1,0 
                    ])
    probe.chanMap = np.array([
                    32,59,12,29,60,13,61,64,48,16,
                    47,14,31,62,15,42,10,27,58,30,63,46,
                    26,24,25,9,43,57,28,45,11,44,
                    # shank 2
                    8,56,40,22,38,7,23,34,18,54,
                    41,55,20,36,6,52,4,35,1,19,39,49,
                    21,5,53,17,33,3,37,50,2,51]) - 1 #its 1 indexed
    probe.chanMap = probe_to_intan[probe.chanMap]

    for shank_id in range(2):
        shank_channels = np.arange(32*shank_id,32*(shank_id + 1))
        # go down in columns
        probe.xc[shank_channels] = np.concatenate((np.ones(10)*0.0,np.ones(12)*18.5,np.ones(10)*2*18.5)) + 150 * shank_id
        probe.yc[shank_channels] = np.concatenate(((45.0 + 30*np.arange(10)),(30*np.arange(12)),(45+30*np.arange(10))))
    return probe