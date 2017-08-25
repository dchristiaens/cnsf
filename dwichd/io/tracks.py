'''
Functions to read and write track files in various formats.

Copyright (c) 2012 - Daan Christiaens (daan.christiaens@gmail.com)
'''

import numpy as np
import sys


def read_tracks(filename):
    if filename.endswith('.tck'):
        return _read_mrtrix_tracks(filename)
    elif filename.endswith('.vtk'):
        return _read_vtk_tracks(filename)
    elif filename.endswith('.fib'):
        return _read_mitk_tracks(filename)
    else:
        raise IOError('File type not supported.')


def write_tracks(filename, tracks):
    if filename.endswith('.tck'):
        _write_mrtrix_tracks(filename, tracks)
    else:
        raise IOError('File type not supported.')


def _read_mrtrix_tracks(filename):
    '''
    Reads MRtrix '.tck' track files.
    '''
    f = open(filename)
    # read header
    fl = ''
    while fl != 'END':
        fl = f.readline().strip()
        if fl.startswith('datatype'):
            dtstr = fl.split(':')[1].strip()
            dt = np.dtype({'Float32': '=f4', 'Float32LE': '<f4', 'Float32BE': '>f4'}[dtstr])
        if fl.startswith('file'):
            offset = int(fl.split('.')[1].strip());
    # read track data
    f.seek(offset, 0)
    rawtracks = np.fromfile(file=f, dtype=dt).reshape((-1,3))
    f.close()
    tracks = np.vsplit(rawtracks, np.where(np.isnan(rawtracks))[0][::3])
    tracks2 = [tracks[0]] + [track[1:,:] for track in tracks[1:-1]]
    return tracks2


def _write_mrtrix_tracks(filename, Tracks):
    '''
    Writes a list of Tracks to the MRtrix '.tck' file format.
    '''
    f = open(filename, 'wb')
    # write header
    f.write('mrtrix tracks\n')
    f.write('timestamp: 0000000000.0000000000\n')
    lbe = 'BE' if (sys.byteorder == 'big') else 'LE'
    f.write('datatype: Float32' + lbe + '\n')
    f.write('count: ' + str(len(Tracks)) + '\n')
    f.write('total_count: ' + str(len(Tracks)) + '\n')
    f.flush()
    offset = f.tell() + 13
    offset += np.floor(np.log10(offset)) + 1
    f.write('file: . ' + str(int(offset)) + '\n')
    f.write('END\n')
    f.flush()
    # write data
    eot = np.empty((3,), dtype=np.float32)
    eot.fill(np.NAN)
    eof = np.empty((3,), dtype=np.float32)
    eof.fill(np.Infinity)
    for t in Tracks:
        t.astype(np.float32).tofile(f)
        eot.tofile(f)
    eof.tofile(f)
    f.close()


def _read_mitk_tracks(filename):
    '''
    Reads MITK '.fib' track files.
    '''
    f = open(filename)
    fl = ''
    while not fl.startswith('POINTS'):
        fl = f.readline().strip()
    N = int(fl.split(' ')[1])
    Points = np.zeros((3*N,), dtype=np.float32)
    k = 0
    while k < 3*N:
        fl = f.readline().strip()
        l = map(float, fl.split(' '))
        Points[k:k+len(l)] = l
        k += len(l)
    Points = Points.reshape((N,3))
    Tracks = []
    f.seek(0)
    fl = ''
    while not fl.startswith('LINES'):
        fl = f.readline().strip()
    M = int(fl.split(' ')[1])
    for k in xrange(M):
        fl = f.readline().strip()
        l = map(int, fl.split(' '))
        Tracks.append(np.copy(Points[l[1:],:]))
    f.close()
    return Tracks


def _read_vtk_tracks(filename):
    '''
    Reads tracks in VTK polygondata.
    '''
    f = open(filename)
    fl = ''
    while not fl.startswith('POINTS'):
        fl = f.readline().strip()
    N = int(fl.split(' ')[1])
    dt = {'float': '>f4', 'double': '>f8'}[fl.split(' ')[2]]
    Points = np.fromfile(f, dtype=dt, count=3*N).reshape((-1,3))
    f.seek(0)
    while not fl.startswith('LINES'):
        fl = f.readline().strip()
    N = int(fl.split(' ')[1])
    M = int(fl.split(' ')[2])
    Lines = np.fromfile(f, dtype='>i4', count=M)
    Tracks = []
    idx = 0
    for k in xrange(N):
        nidx = idx + Lines[idx] + 1
        Tracks.append(Points[Lines[idx+1:nidx],:])
        idx = nidx
    return Tracks


def read_mrtrix_tsf(filename):
    '''
    Reads MRtrix '.tsf' track scalar files.
    '''
    f = open(filename)
    # read header
    fl = ''
    while fl != 'END':
        fl = f.readline().strip()
        if fl.startswith('datatype'):
            dtstr = fl.split(':')[1].strip()
            dt = np.dtype({'Float32': '=f4', 'Float32LE': '<f4', 'Float32BE': '>f4'}[dtstr])
        if fl.startswith('file'):
            offset = int(fl.split('.')[1].strip());
    # read track data
    f.seek(offset, 0)
    data = np.fromfile(file=f, dtype=dt)
    f.close()
    return data[::2]


def write_mrtrix_tsf(filename, scalars):
    '''
    Writes MRtrix '.tsf' track scalar files.
    '''
    f = open(filename, 'wb')
    # write header
    f.write('mrtrix track scalars\n')
    f.write('timestamp: 0000000000.0000000000\n')
    lbe = 'BE' if (sys.byteorder == 'big') else 'LE'
    f.write('datatype: Float32' + lbe + '\n')
    f.write('count: ' + str(len(scalars)) + '\n')
    f.write('total_count: ' + str(len(scalars)) + '\n')
    f.flush()
    offset = f.tell() + 13
    offset += np.floor(np.log10(offset)) + 1
    f.write('file: . ' + str(int(offset)) + '\n')
    f.write('END\n')
    f.flush()
    # write data
    eot = np.empty((3,), dtype=np.float32)  # Fixed to 3 scalars, generalize!
    eot.fill(np.NAN)
    eof = np.empty((3,), dtype=np.float32)
    eof.fill(np.Infinity)
    for t in scalars:
        t.astype(np.float32).tofile(f)
        eot.tofile(f)
    eof.tofile(f)
    f.close()



