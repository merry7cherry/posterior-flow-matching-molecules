"""Plot one generated conformer colored by endpoint posterior covariance trace."""
import argparse
from pathlib import Path
import numpy as np
from etflow.commons.pfm_artifacts import EvaluationRun, run_lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--molecule', type=int, default=0, help='Diagnostic molecule index (0 to 7)')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite an existing figure')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with run_lock(args.run_dir):
        run = EvaluationRun(args.run_dir)
        run.load_complete()
        record = run.read_molecule(args.molecule)
        if record['diagnostic'] is None:
            raise ValueError('No trajectory diagnostic for this molecule; choose an index below 8')
        data, diagnostic = record['data'], record['diagnostic']
        xyz = np.asarray(data.pos_gen[0])
        trace = 3 * diagnostic['frames'][-1]['endpoint_variance'][:len(xyz)].numpy()
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')
        for bond in data.rdmol.GetBonds():
            points = xyz[[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]]
            ax.plot(*points.T, color='0.6', linewidth=2)
        im = ax.scatter(*xyz.T, c=trace, cmap='magma', s=65)
        ax.set_box_aspect([1, 1, 1]); ax.set_axis_off()
        fig.colorbar(im, ax=ax, shrink=.6, label='Endpoint covariance trace (angstrom²)')
        label = 'Generated conformer; model posterior readout'
        if 'postprocessing' in diagnostic:
            label += '\nRaw-model variance after stereo correction'
        ax.set_title(label, fontsize=10)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=160, bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    main()
