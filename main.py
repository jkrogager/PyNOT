"""
Automatically Classify and Reduce a given Data Set
"""

from argparse import ArgumentParser
import os
import datetime

from . import alfosc
from . import data_organizer as do
from .calibs import combine_bias_frames, combine_flat_frames, normalize_spectral_flat

code_dir = os.path.dirname(os.path.abspath(__file__))
calib_dir = os.path.join(code_dir, '/calib/')
v_file = os.path.join(code_dir, 'VERSION')
with open(v_file) as version_file:
    __version__ = version_file.read().strip()


class Report(object):
    def __init__(self):
        self.time = datetime.datetime.now()
        self.fname = 'pynot_%s.log' % self.time.strftime('%d%b%Y-%Hh%Mm%S')
        self.remarks = list()
        self.lines = list()
        self.header = """
        #  PyNOT Data Processing Pipeline
        # ================================
        # version %s
        %s

        """ % (__version__, self.time.strftime("%b %d, %Y  %H:%M:%S"))
        self.report = ""

    def commit(self, text):
        self.lines.append(text)

    def error(self, text):
        text = ' [ERROR]  - ' + text
        if text[-1] != '\n':
            text += '\n'
        self.lines.append(text)

    def warn(self, text):
        text = '[WARNING] - ' + text
        if text[-1] != '\n':
            text += '\n'
        self.lines.append(text)

    def write(self, text):
        text = '          > ' + text
        if text[-1] != '\n':
            text += '\n'
        self.lines.append(text)

    def add_linebreak(self):
        self.lines.append("\n")

    def add_remark(self, text):
        self.remarks.append(text)

    def _make_report(self):
        remark_str = ''.join(self.remarks)
        lines_str = ''.join(self.lines)
        self.report = '\n'.join([self.header, remark_str, lines_str])

    def print_report(self):
        self._make_report()
        print(self.report)

    def save(self):
        self._make_report()
        with open(self.fname, 'w') as output:
            output.write(self.report)


def main(raw_path, verbose=True):
    log = Report()

    if not os.path.exists(raw_path):
        log.error("Data path does not exist : %s" % raw_path)
        log.write("Pipeline terminated!")
        return

    # Classify files:
    log.write("Classyfying files in folder: %s" % raw_path)
    try:
        collection, message = do.classify(raw_path, progress=verbose)
        database = do.TagDatabase(collection)
        log.commit(message)
    except ValueError as err:
        log.error(str(err))
        print(err)
    except FileNotFoundError as err:
        log.error(str(err))
        print(err)
    finally:
        log.save()
        return

    # -- Organize object files in dataset:
    object_filelist = database['SPEC_OBJECT']
    object_images = list(map(do.RawImage, object_filelist))
    if verbose:
        print("\n - The following objects were found in the dataset:")
        print("OBJECT                  GRISM       SLIT      EXPTIME")
        for sci_img in object_images:
            print("%20s  %9s  %11s  %.0f" % (sci_img.object, sci_img.grism, sci_img.slit, sci_img.exptime))
        print("")

    # -- Check BIAS:

    # -- Check FLAT:

    # -- Check FLUX_STD:

    # -- Check ARC:
    arc_images = list()
    for arc_type in ['ARC_He', 'ARC_HeNe', 'ARC_Ne', 'ARC_ThAr']:
        if arc_type in database.keys():
            arc_images += list(map(lambda x: do.RawImage(x, arc_type), database[arc_type]))
    grism_list = list()
    # get list of unique grisms in dataset:
    for sci_img in object_images:
        grism_name = alfosc.grism_translate[sci_img.grism]
        if grism_name not in grism_list:
            grism_list.append(grism_name)

    # Check if pixeltable exists:
    create_pixtab = list()
    for grism_name in grism_list:
        pixtab_fname = calib_dir + '/%s_pixeltable.dat' % grism_name
        if not os.path.exists(pixtab_fname):
            create_pixtab.append(grism_name)

    if len(create_pixtab) == 0:
        pass
    elif len(create_pixtab) == 1:
        print("\n - The following grism is used in the dataset")
        print("   but no pixeltable exists in the calibration database:")
    else:
        print("\n - The following grisms are used in the dataset")
        print("   but no pixeltable exists in the calibration database:")
    for grism_name in create_pixtab:
        print(grism_name)

    print("\n   Starting interactive definition of pixeltable...")
    print("   Get your line identification plot ready!")
    for grism_name in create_pixtab:
        arc_images_for_grism = list()
        for arc_img in arc_images:
            this_grism = alfosc.grism_translate[arc_img.grism]
            if this_grism == grism_name:
                arc_images_for_grism.append(arc_img)

        if len(arc_images_for_grism) == 0:
            print("[ERROR] - No arc frames defined for grism: " + grism_name)
            return None
        else:
            create_pixtable(arc_images_for_grism, grism_name)

    for sci_img in object_images:
        output_dir = sci_img.target_name
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        master_bias_fname = os.path.join(output_dir, 'MASTER_BIAS.fits')
        grism = alfosc.grism_translate[sci_img.grism]
        comb_flat_fname = os.path.join(output_dir, 'FLAT_COMBINED_%s_%s.fits' % (grism, sci_img.slit))
        norm_flat_fname = os.path.join(output_dir, 'NORM_FLAT_%s_%s.fits' % (grism, sci_img.slit))
        final_2d_fname = os.path.join(output_dir, 'red2D_%s_%s.fits' % (sci_img.target_name, sci_img.date))

        # Combine Bias Frames matched for CCD setup
        bias_frames = sci_img.match_files(database['BIAS'])
        combine_bias_frames(bias_frames, output=master_bias_fname,
                            kappa=bias_kappa,
                            verbose=verbose)

        # Combine Flat Frames matched for CCD setup, grism, slit and filter
        flat_frames = sci_img.match_files(database['SPEC_FLAT'], grism=True, slit=True, filter=True)
        _ = combine_flat_frames(flat_frames, mbias=master_bias_fname,
                                output=comb_flat_fname,
                                kappa=flat_kappa, verbose=verbose)

        # Normalize the spectral flat field:
        normalize_spectral_flat(comb_flat_fname, output=norm_flat_fname,
                                axis=dispaxis,
                                x1=args.flat_x1, x2=args.flat_x2,
                                order=args.flat_order, sigma=args.flat_sigma,
                                plot=args.plot, show=args.show, ext=args.ext,
                                clobber=False, verbose=args.verbose)

        # Sensitivity Function:
        std_frames = sci_img.match_files(database['SPEC_FLUX-STD'], grism=True, slit=True, filter=True)

        # Science Reduction:


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("path", type=str,
                        help="Path to directory containing the raw data")
    main(parser.path)
