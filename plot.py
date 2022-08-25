import os
import argparse
import common.plot_util as pu


def main(args):

    try:
        results = pu.load_results(args.log_dir)
    except ValueError:
        print("Either list of directories or a directory incorrect")

    results_dir = 'images/'

    pu.plot_results(results, experiments_name=args.figure_name, results_dir=results_dir, average_group=True, shaded_std=False, legend_outside=args.legend_outside, multiplots=args.multiplots, tiling=args.tiling, figsize=args.figsize, xlabel=args.xlabel, ylabel=args.ylabel)

    print("Plot successfully!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Plot")
    parser.add_argument(
        '--log-dir',
        default='/logs/',
        help='a list of directories of results or a directory')    
    parser.add_argument(
        '--figure-name',
        type=str,
        default="Figure_1",
        help='the name of the figure to be saved')
    parser.add_argument(
        '--legend-outside',
        action='store_true',
        default=False,
        help='set legend outside of the box')
    parser.add_argument(
        '--multiplots',
        action='store_true',
        default=False,
        help='plot multiple subplots')
    parser.add_argument(
        '--figsize',
        default=(8, 6),
        help='figure size')
    parser.add_argument(
        '--tiling',
        type=str,
        default='horizontal',
        help='layout of subplots (default: vertical | horizontal | symmetric')
    parser.add_argument(
        '--xlabel',
        type=str,
        default='Enviroment Steps (millions)',
        help='xaxis label')
    parser.add_argument(
        '--ylabel',
        type=str,
        default='Average Return',
        help='yaxis label')
    

    args = parser.parse_args()
    
    main(args)



