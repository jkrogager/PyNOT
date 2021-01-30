
import multi_extract as me

## Single object
#fname = 'SCI_2D_crr_ALAb170110.fits'
#msg = me.auto_extract(fname, 'output/single_1D.fits', dx=10, dispaxis=2, pdf_fname='output/single_trace.pdf')
#print(msg)
#print()

# Two objects:
fname = 'GQ_raw.fits'
msg = me.auto_extract(fname, 'output/double_1D.fits', pdf_fname='output/double_trace.pdf', width_order=0)
print(msg)

# Flux Standard:
fname = '/Users/krogager/coding/PyNOT/test/fluxstd_raw.fits'
msg = me.auto_extract(fname, 'output/std_1D.fits', N=1, model_name='tophat',
                      pdf_fname='output/std_trace.pdf', dx=10, width_order=5)
print(msg)

