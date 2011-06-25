/*
 *  ***** BEGIN LICENSE BLOCK ***** Version: MPL 1.1/GPL 2.0/LGPL 2.1
 * 
 * The contents of this file are subject to the Mozilla Public License Version
 * 1.1 (the "License"); you may not use this file except in compliance with the
 * License. You may obtain a copy of the License at http://www.mozilla.org/MPL/
 * 
 * Software distributed under the License is distributed on an "AS IS" basis,
 * WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License for
 * the specific language governing rights and limitations under the License.
 * 
 * The Original Code is JTransforms.
 * 
 * The Initial Developer of the Original Code is Piotr Wendykier, Emory
 * University. Portions created by the Initial Developer are Copyright (C)
 * 2007-2009 the Initial Developer. All Rights Reserved.
 * 
 * Alternatively, the contents of this file may be used under the terms of
 * either the GNU General Public License Version 2 or later (the "GPL"), or the
 * GNU Lesser General Public License Version 2.1 or later (the "LGPL"), in which
 * case the provisions of the GPL or the LGPL are applicable instead of those
 * above. If you wish to allow use of your version of this file only under the
 * terms of either the GPL or the LGPL, and not to allow others to use your
 * version of this file under the terms of the MPL, indicate your decision by
 * deleting the provisions above and replace them with the notice and other
 * provisions required by the GPL or the LGPL. If you do not delete the
 * provisions above, a recipient may use your version of this file under the
 * terms of any one of the MPL, the GPL or the LGPL.
 * 
 * ***** END LICENSE BLOCK *****
 */

package edu.emory.mathcs.jtransforms.fft;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Random;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Test of the utility class {@link RealFFTUtils_2D}.
 * 
 * @author S&eacute;bastien Brisard
 * 
 */
@RunWith(value = Parameterized.class)
public class RealFFTUtils_2DTest {
    /** The constant value of the seed of the random generator. */
    public static final int SEED = 20110624;

    @Parameters
    public static Collection<Object[]> getParameters() {
        final int[] size = { 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 };

        final ArrayList<Object[]> parameters = new ArrayList<Object[]>();

        for (int i = 0; i < size.length; i++) {
            for (int j = 0; j < size.length; j++) {
                parameters.add(new Object[] { size[i], size[j], 1, 1, SEED });
                parameters.add(new Object[] { size[i], size[j], 4, 1, SEED });
            }
        }
        return parameters;
    }

    /** Number of columns of the data arrays to be Fourier transformed. */
    private final int columns;

    /** To perform FFTs on double precision data. */
    private final DoubleFFT_2D fft2d;

    /** To perform FFTs on single precision data. */
    private final FloatFFT_2D fft2f;

    /**
     * Specified accuracy for equality checks. The equality check reads <center>
     * <code>norm(actual - expected) < maxUlps * Math.ulp(norm(expected))</code>
     * , </center> where norm is the L2-norm (euclidean).
     */
    private final int maxUlps;

    /** For the generation of the data arrays. */
    private final Random random;

    /** Number of rows of the data arrays to be Fourier transformed. */
    private final int rows;

    /** The object to be tested. */
    private final RealFFTUtils_2D unpacker;

    /**
     * Creates a new instance of this test.
     * 
     * @param rows
     *            number of rows
     * @param columns
     *            number of columns
     * @param numThreads
     *            the number of threads to be used
     * @param maxUlps
     *            see {@link #maxUlps}
     * @param seed
     *            the seed of the random generator
     */
    public RealFFTUtils_2DTest(final int rows, final int columns,
            final int numThreads, final int maxUlps, final long seed) {
        this.rows = rows;
        this.columns = columns;
        this.maxUlps = maxUlps;
        this.fft2d = new DoubleFFT_2D(rows, columns);
        this.fft2f = new FloatFFT_2D(rows, columns);
        this.random = new Random(seed);
        this.unpacker = new RealFFTUtils_2D(rows, columns);
        ConcurrencyUtils.setNumberOfThreads(numThreads);
    }

    /**
     * Throws an exception if the error is too large.
     * 
     * @param val
     *            expected value
     * @param err
     *            error on the expected value
     */
    public void checkRelativeError(final double val, final double err) {
        final double numUlps = err / Math.ulp(val);
        Assert.assertTrue("2d-FFT of size " + rows + "x" + columns
                + ", exp. rel. err. = " + maxUlps + " ulps, act. rel. err. = "
                + numUlps + " ulps", numUlps <= maxUlps);
    }

    /**
     * Throws an exception if the error is too large.
     * 
     * @param val
     *            expected value
     * @param err
     *            error on the expected value
     */
    public void checkRelativeError(final float val, final float err) {
        final double numUlps = err / Math.ulp(val);
        Assert.assertTrue("2d-FFT of size " + rows + "x" + columns
                + ", exp. rel. err. = " + maxUlps + " ulps, act. rel. err. = "
                + numUlps + " ulps", numUlps <= maxUlps);
    }

    @Test
    public void testPack1dInput() {
        final double[] actual = new double[rows * columns];
        final double[][] expected = new double[rows][columns];
        final double[][] dummy = new double[rows][2 * columns];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                expected[r][c] = random.nextDouble();
                dummy[r][2 * c] = expected[r][c];
                dummy[r][2 * c + 1] = 0.0;
            }
        }
        fft2d.complexForward(dummy);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < 2 * columns; c++) {
                try {
                    unpacker.pack(dummy[r][c], r, c, actual, 0);
                } catch (IllegalArgumentException e) {
                    // Do nothing
                }
            }
        }
        fft2d.realInverse(actual, true);

        double exp, act, diff;
        double diffNorm = 0.;
        double expNorm = 0.;

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                exp = expected[r][c];
                act = actual[columns * r + c];
                diff = act - exp;
                expNorm += exp * exp;
                diffNorm += diff * diff;
            }
        }
        checkRelativeError(expNorm, diffNorm);
    }

    @Test
    public void testPack1fInput() {
        final float[] actual = new float[rows * columns];
        final float[][] expected = new float[rows][columns];
        final float[][] dummy = new float[rows][2 * columns];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                expected[r][c] = random.nextFloat();
                dummy[r][2 * c] = expected[r][c];
                dummy[r][2 * c + 1] = 0f;
            }
        }
        fft2f.complexForward(dummy);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < 2 * columns; c++) {
                try {
                    unpacker.pack(dummy[r][c], r, c, actual, 0);
                } catch (IllegalArgumentException e) {
                    // Do nothing
                }
            }
        }
        fft2f.realInverse(actual, true);

        float exp, act, diff;
        float diffNorm = 0f;
        float expNorm = 0f;

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                exp = expected[r][c];
                act = actual[columns * r + c];
                diff = act - exp;
                expNorm += exp * exp;
                diffNorm += diff * diff;
            }
        }
        checkRelativeError(expNorm, diffNorm);
    }

    @Test
    public void testPack2dInput() {
        final double[][] actual = new double[rows][columns];
        final double[][] expected = new double[rows][columns];
        final double[][] dummy = new double[rows][2 * columns];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                expected[r][c] = random.nextDouble();
                dummy[r][2 * c] = expected[r][c];
                dummy[r][2 * c + 1] = 0.0;
            }
        }
        fft2d.complexForward(dummy);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < 2 * columns; c++) {
                try {
                    unpacker.pack(dummy[r][c], r, c, actual);
                } catch (IllegalArgumentException e) {
                    // Do nothing
                }
            }
        }
        fft2d.realInverse(actual, true);

        double exp, act, diff;
        double diffNorm = 0.;
        double expNorm = 0.;

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                exp = expected[r][c];
                act = actual[r][c];
                diff = act - exp;
                expNorm += exp * exp;
                diffNorm += diff * diff;
            }
        }
        checkRelativeError(expNorm, diffNorm);
    }

    @Test
    public void testPack2fInput() {
        final float[][] actual = new float[rows][columns];
        final float[][] expected = new float[rows][columns];
        final float[][] dummy = new float[rows][2 * columns];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                expected[r][c] = random.nextFloat();
                dummy[r][2 * c] = expected[r][c];
                dummy[r][2 * c + 1] = 0f;
            }
        }
        fft2f.complexForward(dummy);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < 2 * columns; c++) {
                try {
                    unpacker.pack(dummy[r][c], r, c, actual);
                } catch (IllegalArgumentException e) {
                    // Do nothing
                }
            }
        }
        fft2f.realInverse(actual, true);

        float exp, act, diff;
        float diffNorm = 0f;
        float expNorm = 0f;

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                exp = expected[r][c];
                act = actual[r][c];
                diff = act - exp;
                expNorm += exp * exp;
                diffNorm += diff * diff;
            }
        }
        checkRelativeError(expNorm, diffNorm);
    }

    @Test
    public void testUnpack1dInput() {
        final double[] actual = new double[rows * columns];
        final double[][] expected = new double[rows][2 * columns];
        for (int r = 0; r < rows; r++) {
            final int offset = r * columns;
            for (int c = 0; c < columns; c++) {
                actual[offset + c] = random.nextDouble();
                expected[r][2 * c] = actual[offset + c];
                expected[r][2 * c + 1] = 0.0;
            }
        }
        fft2d.complexForward(expected);
        fft2d.realForward(actual);

        double exp, act, diff;
        double diffNorm = 0.;
        double expNorm = 0.;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < 2 * columns; c++) {
                exp = expected[r][c];
                act = unpacker.unpack(r, c, actual, 0);
                diff = act - exp;
                expNorm += exp * exp;
                diffNorm += diff * diff;
            }
        }
        checkRelativeError(expNorm, diffNorm);
    }

    @Test
    public void testUnpack1fInput() {
        final float[] actual = new float[rows * columns];
        final float[][] expected = new float[rows][2 * columns];
        for (int r = 0; r < rows; r++) {
            final int offset = r * columns;
            for (int c = 0; c < columns; c++) {
                actual[offset + c] = random.nextFloat();
                expected[r][2 * c] = actual[offset + c];
                expected[r][2 * c + 1] = 0f;
            }
        }
        fft2f.complexForward(expected);
        fft2f.realForward(actual);

        float exp, act, diff;
        float diffNorm = 0f;
        float expNorm = 0f;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < 2 * columns; c++) {
                exp = expected[r][c];
                act = unpacker.unpack(r, c, actual, 0);
                diff = act - exp;
                expNorm += exp * exp;
                diffNorm += diff * diff;
            }
        }
        checkRelativeError(expNorm, diffNorm);
    }

    @Test
    public void testUnpack2dInput() {
        final double[][] actual = new double[rows][columns];
        final double[][] expected = new double[rows][2 * columns];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                actual[r][c] = random.nextDouble();
                expected[r][2 * c] = actual[r][c];
                expected[r][2 * c + 1] = 0.0;
            }
        }
        fft2d.complexForward(expected);
        fft2d.realForward(actual);

        double exp, act, diff;
        double diffNorm = 0.;
        double expNorm = 0.;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < 2 * columns; c++) {
                exp = expected[r][c];
                act = unpacker.unpack(r, c, actual);
                diff = act - exp;
                expNorm += exp * exp;
                diffNorm += diff * diff;
            }
        }
        checkRelativeError(expNorm, diffNorm);
    }

    @Test
    public void testUnpack2fInput() {
        final float[][] actual = new float[rows][columns];
        final float[][] expected = new float[rows][2 * columns];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                actual[r][c] = random.nextFloat();
                expected[r][2 * c] = actual[r][c];
                expected[r][2 * c + 1] = 0f;
            }
        }
        fft2f.complexForward(expected);
        fft2f.realForward(actual);

        float exp, act, diff;
        float diffNorm = 0f;
        float expNorm = 0f;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < 2 * columns; c++) {
                exp = expected[r][c];
                act = unpacker.unpack(r, c, actual);
                diff = act - exp;
                expNorm += exp * exp;
                diffNorm += diff * diff;
            }
        }
        checkRelativeError(expNorm, diffNorm);
    }
}
