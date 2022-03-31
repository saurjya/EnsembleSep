import argparse
import json
import yaml
import os
import numpy as np
import scipy
import librosa
import soundfile as sf
import itertools


def make_processed_filelist(track_list, out_path, segment, poly):
    """
    Given list of audio files, generates activity confidence array
    Writes audio file path and confidence array to json file

    Parameters
    ----------
    track_list : list
        List of audio file paths
    out_dir: str
        Output directory to save json
    out_filename : str
        file name for json file
    """
    file_infos = []
    for track in track_list: 
        counter = 0
        print("Processing file", track)
        pairs = get_sets(track,segment,hop_length=2048,N=poly,threshold=0.1)
        if pairs:
            file_infos.extend(pairs)
        counter += 1
    
    print("writing to json file", out_path)
    with open(os.path.join(out_path), "w") as f:
        json.dump(file_infos, f, indent=4)
    print("json file writing complete")
    return


def get_sets(match_files,segment,hop_length,N,threshold):
    activations = []
    pairs = []
    for track in match_files:
        activations.append(compute_activation_confidence(track, 2 * hop_length))
        if activations[-1].size != activations[0].size:
            return False
    files_activations = list(map(lambda x, y:(x,y), match_files, activations))
    file_pairs = list(itertools.combinations(files_activations, N))
    for pair in file_pairs:
        overlap = []
        files = []
        actList = []
        for member in pair:
            actList.append(np.squeeze(member[1]))
            files.append(member[0])
        act = np.array(actList)
        overlap = np.prod(act, axis=0)
        for i in range((len(overlap)//(segment//hop_length))-1):
            step = segment//hop_length;
            conf = overlap[i*step:(i+1)*step].sum()
            #print(conf)
            if conf > step*threshold:
                temp = []
                temp.extend(files)
                temp.append(i*segment)
                pairs.append(temp)
                #print('happen')
    print(len(pairs))
    return pairs


def preprocess_directory(
    data_path,
    inst_list,
    mix_list,
):
    """ Generates list of tracks for each song for given set if instruments. 

    Parameters
    ----------
    data_path : str
        Path containing BBCSO data folders
    inst_list: list
        List of instrument tags to use to filter RAW/STEM tracks
    mix_list : list
        List of mix/mic tags
    
 
    Returns
    -------
    song_tracks : list
        List of set of instrument tracks for each song belonging to given instrument tag list

    """
    counter = 0
    song_tracks = []
    data_path = os.path.abspath(data_path)
    song_list = os.listdir(data_path)
    for song_folder in song_list:
        song_path = os.path.join(data_path,song_folder)
        if not os.path.isdir(song_path):
            continue
        trackmix_list = os.listdir(song_path)
        for folder in trackmix_list:
            for mix in mix_list:
                if mix in folder:
                    path = os.path.join(song_path,folder)
                    filelist = os.listdir(path)
                    data_list = []
                    for file in filelist:
                        for instrument in inst_list:
                            if instrument in file:
                                inst_path = os.path.join(path,file)
                                data_list.append(inst_path)
                    if len(data_list):
                        song_tracks.append(list(data_list))
    return song_tracks


def compute_activation_confidence(
    track, win_len, lpf_cutoff=0.075, theta=0.15, var_lambda=20.0, amplitude_threshold=0.01
):
    """Create the activation confidence annotation for a multitrack. The final
    activation matrix is computed as:
        `C[i, t] = 1 - (1 / (1 + e**(var_lambda * (H[i, t] - theta))))`
    where H[i, t] is the energy of stem `i` at time `t`

    Parameters
    ----------
    track : Audio path
    win_len : int, default=4096
        Number of samples in each window
    lpf_cutoff : float, default=0.075
        Lowpass frequency cutoff fraction
    theta : float
        Controls the threshold of activation.
    var_labmda : float
        Controls the slope of the threshold function.
    amplitude_threshold : float
        Energies below this value are set to 0.0

    Returns
    -------
    C : np.array
        Array of activation confidence values shape (n_conf, n_stems)
    stem_index_list : list
        List of stem indices in the order they appear in C

    """
    H = []
    # MATLAB equivalent to @hanning(win_len)
    win = scipy.signal.windows.hann(win_len + 2)[1:-1]

    # audio, rate = librosa.load(track, mono=True)
    try:
        audio, rate = sf.read(track)
    except:
        return 
    try:
        audio = audio.mean(axis=1)
    except:
        print("mono file")
        
    H.append(track_energy(audio.T, win_len, win))

    # list to numpy array
    H = np.array(H)

    # normalization (to overall energy and # of sources)
    E0 = np.sum(H, axis=0)

    H = H / np.max(E0)
    # binary thresholding for low overall energy events
    H[:, E0 < amplitude_threshold] = 0.0

    # LP filter
    b, a = scipy.signal.butter(2, lpf_cutoff, "low")
    H = scipy.signal.filtfilt(b, a, H, axis=1)

    # logistic function to semi-binarize the output; confidence value
    C = 1.0 - (1.0 / (1.0 + np.exp(np.dot(var_lambda, (H - theta)))))

    # add time column
    time = librosa.core.frames_to_time(np.arange(C.shape[1]), sr=rate, hop_length=win_len // 2)

    # stack time column to matrix
    C_out = np.vstack((time, C))
    # print(C_out.T)
    #return C_out.T, rate
    return C


def track_energy(wave, win_len, win):
    """Compute the energy of an audio signal

    Parameters
    ----------
    wave : np.array
        The signal from which to compute energy
    win_len: int
        The number of samples to use in energy computation
    win : np.array
        The windowing function to use in energy computation

    Returns
    -------
    energy : np.array
        Array of track energy

    """
    hop_len = win_len // 2

    wave = np.lib.pad(wave, pad_width=(win_len - hop_len, 0), mode="constant", constant_values=0)

    # post padding
    wave = librosa.util.fix_length(wave, int(win_len * np.ceil(len(wave) / win_len)))

    # cut into frames
    wavmat = librosa.util.frame(wave, frame_length=win_len, hop_length=hop_len)

    # Envelope follower
    wavmat = hwr(wavmat) ** 0.5  # half-wave rectification + compression

    return np.mean((wavmat.T * win), axis=1)


def hwr(x):
    """ Half-wave rectification.

    Parameters
    ----------
    x : array-like
        Array to half-wave rectify

    Returns
    -------
    x_hwr : array-like
        Half-wave rectified array

    """
    return (x + np.abs(x)) / 2



if __name__ == "__main__":
    """
    To test metadata parsing and confidence array generation
    """
    parser = argparse.ArgumentParser("MedleyDB data preprocessing")
    parser.add_argument(
        "--data_path", type=str, default="/bask/projects/v/vjgo8416-mus-sep/BBCSO/EnsembleSep/AUDIO/ENS", help="Directory path of BBCSO tracks"
    )
    
    parser.add_argument(
        "--inst_list",
        nargs="+",
        help="list of instruments",
        #default=["Violin 1", "Violin 2", "Violin 3", "Cello", "Bass.", "Viola"],
        default=["Violin", "Viola","Cello"],
    )
    parser.add_argument(
        "--mix_list",         nargs="+",
        help="list of mixes",
        default=["CloseW"],
    )
    parser.add_argument(
        "--json_path", type=str, default="/bask/projects/v/vjgo8416-mus-sep/Experiments/vvcFull_train.json", help="Directory path for output json files"
    )
    parser.add_argument(
        "--segment", type=int, default=220500, help="Length of segments in seconds"
    )
    

    args = parser.parse_args()
    print(args)
    tracklist = preprocess_directory(
        args.data_path,
        args.inst_list,
        args.mix_list,
    )
    make_processed_filelist(
        tracklist, 
        args.json_path,
        args.segment,
        poly=2
    )
    