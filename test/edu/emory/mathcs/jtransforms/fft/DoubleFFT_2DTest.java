/* ***** BEGIN LICENSE BLOCK *****
 * Version: MPL 1.1/GPL 2.0/LGPL 2.1
 *
 * The contents of this file are subject to the Mozilla Public License Version
 * 1.1 (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 * http://www.mozilla.org/MPL/
 *
 * Software distributed under the License is distributed on an "AS IS" basis,
 * WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
 * for the specific language governing rights and limitations under the
 * License.
 *
 * The Original Code is JTransforms.
 *
 * The Initial Developer of the Original Code is
 * Piotr Wendykier, Emory University.
 * Portions created by the Initial Developer are Copyright (C) 2007-2009
 * the Initial Developer. All Rights Reserved.
 *
 * Alternatively, the contents of this file may be used under the terms of
 * either the GNU General Public License Version 2 or later (the "GPL"), or
 * the GNU Lesser General Public License Version 2.1 or later (the "LGPL"),
 * in which case the provisions of the GPL or the LGPL are applicable instead
 * of those above. If you wish to allow use of your version of this file only
 * under the terms of either the GPL or the LGPL, and not to allow others to
 * use your version of this file under the terms of the MPL, indicate your
 * decision by deleting the provisions above and replace them with the notice
 * and other provisions required by the GPL or the LGPL. If you do not delete
 * the provisions above, a recipient may use your version of this file under
 * the terms of any one of the MPL, the GPL or the LGPL.
 *
 * ***** END LICENSE BLOCK ***** */

package edu.emory.mathcs.jtransforms.fft;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Random;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import edu.emory.mathcs.jtransforms.fft.DoubleFFT_1D;
import edu.emory.mathcs.jtransforms.fft.DoubleFFT_2D;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * <p>
 * This is a test of the class {@link DoubleFFT_2D}. In this test, a very crude
 * 2d FFT method is implemented (see {@link #complexForward(double[][])}),
 * assuming that {@link DoubleFFT_1D} has been fully tested and validated. This
 * crude (unoptimized) method is then used to establish <em>expected</em> values
 * of <em>direct</em> Fourier transforms.
 * </p>
 * <p>
 * For <em>inverse</em> Fourier transforms, the test assumes that the
 * corresponding <em>direct</em> Fourier transform has been tested and
 * validated.
 * </p>
 * <p>
 * In all cases, the test consists in creating a random array of data, and
 * verifying that expected and actual values of its Fourier transform coincide
 * (L2 norm is zero, within a specified accuracy).
 * </p>
 * 
 * @author S&eacute;bastien Brisard
 * 
 */
@RunWith(value = Parameterized.class)
public class DoubleFFT_2DTest {
    @Parameters
    public static Collection<Object[]> getParameters() {
        final int[] size = { 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 32,
                64, 100, 120, 128, 256, 310, 511, 512, 1024 };

        final ArrayList<Object[]> parameters = new ArrayList<Object[]>();

        for (int i = 0; i < size.length; i++) {
            for (int j = 0; j < size.length; j++) {
                parameters
                        .add(new Object[] { size[i], size[j], 1, 15, 20110602 });
                parameters
                        .add(new Object[] { size[i], size[j], 4, 15, 20110602 });
            }
        }
        return parameters;
    }

    /**
     * Fourier transform of the columns.
     */
    private final DoubleFFT_1D cfft;

    /**
     * The object to be tested.
     */
    private final DoubleFFT_2D fft;

    /**
     * Specified accuracy for equality checks. The equality check reads <center>
     * <code>norm(actual - expected) < maxUlps * Math.ulp(norm(expected))</code>
     * , </center> where norm is the L2-norm (euclidean).
     */
    private final int maxUlps;

    /**
     * Number of columns of the data arrays to be Fourier transformed.
     */
    private final int numCols;

    /**
     * Number of rows of the data arrays to be Fourier transformed.
     */
    private final int numRows;

    /**
     * Fourier transform of the rows.
     */
    private final DoubleFFT_1D rfft;

    /**
     * For the generation of the data arrays.
     */
    private final Random random;

    /**
     * Creates a new instance of this test.
     * 
     * @param numRows
     *            number of rows
     * @param numColumns
     *            number of columns
     * @param numThreads
     *            the number of threads to be used
     * @param maxUlps
     *            see {@link #maxUlps}
     * @param seed
     *            the seed of the random generator
     */
    public DoubleFFT_2DTest(final int numRows, final int numColumns,
            final int numThreads, final int maxUlps, final long seed) {
        this.numRows = numRows;
        this.numCols = numColumns;
        this.maxUlps = maxUlps;
        this.rfft = new DoubleFFT_1D(numColumns);
        this.cfft = new DoubleFFT_1D(numRows);
        this.fft = new DoubleFFT_2D(numRows, numColumns);
        this.random = new Random(seed);
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
        Assert.assertTrue("2d-FFT of size " + numRows + "x" + numCols
                + ", exp. rel. err. = " + maxUlps + " ulps, act. rel. err. = "
                + numUlps + " ulps", numUlps <= maxUlps);
    }

    /**
     * A crude implementation of 2d complex FFT.
     * 
     * @param a
     *            the data to be transformed
     */
    public void complexForward(final double[][] a) {
        for (int r = 0; r < numRows; r++) {
            rfft.complexForward(a[r]);
        }
        final double[] buffer = new double[2 * numRows];
        for (int c = 0; c < numCols; c++) {
            for (int r = 0; r < numRows; r++) {
                buffer[2 * r] = a[r][2 * c];
                buffer[2 * r + 1] = a[r][2 * c + 1];
            }
            cfft.complexForward(buffer);
            for (int r = 0; r < numRows; r++) {
                a[r][2 * c] = buffer[2 * r];
                a[r][2 * c + 1] = buffer[2 * r + 1];
            }
        }
    }

    /**
     * A test of {@link DoubleFFT_2D#complexForward(double[])}.
     */
    @Test
    public void testComplexForward1dInput() {
        final double[] actual = new double[2 * numRows * numCols];
        final double[][] expected = new double[numRows][2 * numCols];
        for (int r = 0; r < numRows; r++) {
            final double[] rexp = expected[r];
            final int offset = 2 * r * numCols;
            for (int c = 0; c < 2 * numCols; c++) {
                actual[offset + c] = random.nextDouble();
                rexp[c] = actual[offset + c];
            }
        }
        fft.complexForward(actual);
        complexForward(expected);
        double diffNorm = 0.;
        double expNorm = 0.;
        for (int r = 0; r < numRows; r++) {
            final double[] rexp = expected[r];
            final int offset = 2 * r * numCols;
            for (int c = 0; c < 2 * numCols; c++) {
                final double exp = rexp[c];
                final double act = actual[offset + c];
                final double diff = act - exp;
                expNorm += exp * exp;
                diffNorm += diff * diff;
            }
        }
        expNorm = Math.sqrt(expNorm);
        diffNorm = Math.sqrt(diffNorm);
        checkRelativeError(expNorm, diffNorm);
    }

    /**
     * A test of {@link DoubleFFT_2D#complexForward(double[][])}.
     */
    @Test
    public void testComplexForward2dInput() {
        final double[][] actual = new double[numRows][2 * numCols];
        final double[][] expected = new double[numRows][2 * numCols];
        for (int r = 0; r < numRows; r++) {
            final double[] rexp = expected[r];
            final double[] ract = actual[r];
            for (int c = 0; c < 2 * numCols; c++) {
                ract[c] = random.nextDouble();
                rexp[c] = ract[c];
            }
        }
        fft.complexForward(actual);
        complexForward(expected);
        double diffNorm = 0.;
        double expNorm = 0.;
        for (int r = 0; r < numRows; r++) {
            final double[] rexp = expected[r];
            final double[] ract = actual[r];
            for (int c = 0; c < 2 * numCols; c++) {
                final double exp = rexp[c];
                final double act = ract[c];
                final double diff = act - exp;
                expNorm += exp * exp;
                diffNorm += diff * diff;
            }
        }
        expNorm = Math.sqrt(expNorm);
        diffNorm = Math.sqrt(diffNorm);
        checkRelativeError(expNorm, diffNorm);
    }

    /**
     * A test of {@link DoubleFFT_2D#complexInverse(double[], boolean)}, with
     * the second parameter set to <code>true</code>.
     */
    @Test
    public void testComplexInverseScaled1dInput() {
        final double[] expected = new double[2 * numRows * numCols];
        final double[] actual = new double[2 * numRows * numCols];
        for (int i = 0; i < actual.length; i++) {
            actual[i] = random.nextDouble();
            expected[i] = actual[i];
        }
        fft.complexForward(actual);
        fft.complexInverse(actual, true);
        double diffNorm = 0.;
        double expNorm = 0.;
        for (int i = 0; i < actual.length; i++) {
            final double exp = expected[i];
            final double act = actual[i];
            final double diff = act - exp;
            expNorm += exp * exp;
            diffNorm += diff * diff;
        }
        expNorm = Math.sqrt(expNorm);
        diffNorm = Math.sqrt(diffNorm);
        checkRelativeError(expNorm, diffNorm);
    }

    /**
     * A test of {@link DoubleFFT_2D#complexInverse(double[][], boolean)}, with
     * the second parameter set to <code>true</code>.
     */
    @Test
    public void testComplexInverseScaled2dInput() {
        final double[][] expected = new double[numRows][2 * numCols];
        final double[][] actual = new double[numRows][2 * numCols];
        for (int r = 0; r < numRows; r++) {
            final double[] rexp = expected[r];
            final double[] ract = actual[r];
            for (int c = 0; c < 2 * numCols; c++) {
                ract[c] = random.nextDouble();
                rexp[c] = ract[c];
            }
        }
        fft.complexForward(actual);
        fft.complexInverse(actual, true);
        double diffNorm = 0.;
        double expNorm = 0.;
        for (int r = 0; r < numRows; r++) {
            final double[] rexp = expected[r];
            final double[] ract = actual[r];
            for (int c = 0; c < 2 * numCols; c++) {
                final double exp = rexp[c];
                final double act = ract[c];
                final double diff = act - exp;
                expNorm += exp * exp;
                diffNorm += diff * diff;
            }
        }
        expNorm = Math.sqrt(expNorm);
        diffNorm = Math.sqrt(diffNorm);
        checkRelativeError(expNorm, diffNorm);
    }

    /**
     * A test of {@link DoubleFFT_2D#complexInverse(double[], boolean)}, with
     * the second parameter set to <code>false</code>.
     */
    @Test
    public void testComplexInverseUnScaled1dInput() {
        final double[] expected = new double[2 * numRows * numCols];
        final double[] actual = new double[2 * numRows * numCols];
        for (int i = 0; i < actual.length; i++) {
            actual[i] = random.nextDouble();
            expected[i] = actual[i];
        }
        fft.complexForward(actual);
        fft.complexInverse(actual, false);
        double diffNorm = 0.;
        double expNorm = 0.;
        final double s = numRows * numCols;
        for (int i = 0; i < actual.length; i++) {
            final double exp = s * expected[i];
            final double act = actual[i];
            final double diff = act - exp;
            expNorm += exp * exp;
            diffNorm += diff * diff;
        }
        expNorm = Math.sqrt(expNorm);
        diffNorm = Math.sqrt(diffNorm);
        checkRelativeError(expNorm, diffNorm);
    }

    /**
     * A test of {@link DoubleFFT_2D#complexInverse(double[][], boolean)}, with
     * the second parameter set to <code>false</code>.
     */
    @Test
    public void testComplexInverseUnScaled2dInput() {
        final double[][] expected = new double[numRows][2 * numCols];
        final double[][] actual = new double[numRows][2 * numCols];
        for (int r = 0; r < numRows; r++) {
            final double[] rexp = expected[r];
            final double[] ract = actual[r];
            for (int c = 0; c < 2 * numCols; c++) {
                ract[c] = random.nextDouble();
                rexp[c] = ract[c];
            }
        }
        fft.complexForward(actual);
        fft.complexInverse(actual, false);
        double diffNorm = 0.;
        double expNorm = 0.;
        final double s = numRows * numCols;
        for (int r = 0; r < numRows; r++) {
            final double[] rexp = expected[r];
            final double[] ract = actual[r];
            for (int c = 0; c < 2 * numCols; c++) {
                final double exp = s * rexp[c];
                final double act = ract[c];
                final double diff = act - exp;
                expNorm += exp * exp;
                diffNorm += diff * diff;
            }
        }
        expNorm = Math.sqrt(expNorm);
        diffNorm = Math.sqrt(diffNorm);
        checkRelativeError(expNorm, diffNorm);
    }

    /**
     * A test of {@link DoubleFFT_2D#realForward(double[])}.
     */
    @Test
    public void testRealForward1dInput() {
        if ((!ConcurrencyUtils.isPowerOf2(numRows))
                || (!ConcurrencyUtils.isPowerOf2(numCols))) {
            return;
        }
        final double[] actual = new double[numRows * numCols];
        final double[][] expected = new double[numRows][2 * numCols];
        final boolean[] checked = new boolean[numRows * numCols];
        Arrays.fill(checked, false);
        for (int r = 0; r < numRows; r++) {
            final double[] rexp = expected[r];
            final int offset = r * numCols;
            for (int c = 0; c < numCols; c++) {
                actual[offset + c] = random.nextDouble();
                rexp[2 * c] = actual[offset + c];
                rexp[2 * c + 1] = 0.0;
            }
        }
        fft.realForward(actual);
        complexForward(expected);
        double exp, act, diff;
        double[] rexp;
        double diffNorm = 0.;
        double expNorm = 0.;
        for (int r = 1; r < numRows; r++) {
            rexp = expected[r];
            final int offset = r * numCols;
            for (int c = 2; c < numCols; c++) {
                exp = rexp[c];
                act = actual[offset + c];
                diff = act - exp;
                expNorm += exp * exp;
                diffNorm += diff * diff;
                checked[offset + c] = true;
            }
        }

        rexp = expected[0];
        for (int c = 2; c < numCols; c++) {
            exp = rexp[c];
            act = actual[c];
            diff = act - exp;
            expNorm += exp * exp;
            diffNorm += diff * diff;
            checked[c] = true;
        }

        for (int r = 1; r < numRows / 2; r++) {
            exp = expected[r][0];
            act = actual[r * numCols];
            diff = act - exp;
            expNorm += exp * exp;
            diffNorm += diff * diff;
            checked[r * numCols] = true;

            exp = expected[r][1];
            act = actual[r * numCols + 1];
            diff = act - exp;
            expNorm += exp * exp;
            diffNorm += diff * diff;
            checked[r * numCols + 1] = true;

            exp = expected[numRows - r][numCols];
            act = actual[(numRows - r) * numCols + 1];
            diff = act - exp;
            expNorm += exp * exp;
            diffNorm += diff * diff;
            checked[(numRows - r) * numCols + 1] = true;

            exp = expected[numRows - r][numCols + 1];
            act = actual[(numRows - r) * numCols];
            diff = act - exp;
            expNorm += exp * exp;
            diffNorm += diff * diff;
            checked[(numRows - r) * numCols] = true;
        }

        exp = expected[0][0];
        act = actual[0];
        diff = act - exp;
        expNorm += exp * exp;
        diffNorm += diff * diff;
        checked[0] = true;

        exp = expected[0][numCols];
        act = actual[1];
        diff = act - exp;
        expNorm += exp * exp;
        diffNorm += diff * diff;
        checked[1] = true;

        exp = expected[numRows / 2][0];
        act = actual[(numRows / 2) * numCols];
        diff = act - exp;
        expNorm += exp * exp;
        diffNorm += diff * diff;
        checked[(numRows / 2) * numCols] = true;

        exp = expected[numRows / 2][numCols];
        act = actual[(numRows / 2) * numCols + 1];
        diff = act - exp;
        expNorm += exp * exp;
        diffNorm += diff * diff;
        checked[(numRows / 2) * numCols + 1] = true;

        expNorm = Math.sqrt(expNorm);
        diffNorm = Math.sqrt(diffNorm);
        for (int r = 0; r < numRows; r++) {
            for (int c = 0; c < numCols; c++) {
                Assert.assertTrue(String.format("[%d][%d]", r, c), checked[r
                        * numCols + c]);
            }
        }
        checkRelativeError(expNorm, diffNorm);
    }

    /**
     * A test of {@link DoubleFFT_2D#realForward(double[][])}.
     */
    @Test
    public void testRealForward2dInput() {
        if ((!ConcurrencyUtils.isPowerOf2(numRows))
                || (!ConcurrencyUtils.isPowerOf2(numCols))) {
            return;
        }
        final double[][] actual = new double[numRows][numCols];
        final double[][] expected = new double[numRows][2 * numCols];
        final boolean[][] checked = new boolean[numRows][numCols];
        for (int r = 0; r < numRows; r++) {
            final double[] rexp = expected[r];
            final double[] ract = actual[r];
            final boolean[] rchecked = checked[r];
            for (int c = 0; c < numCols; c++) {
                ract[c] = random.nextDouble();
                rexp[2 * c] = ract[c];
                rexp[2 * c + 1] = 0.0;
                rchecked[c] = false;
            }
        }
        fft.realForward(actual);
        complexForward(expected);
        double exp, act, diff;
        boolean[] rchecked;
        double[] rexp, ract;
        double diffNorm = 0.;
        double expNorm = 0.;

        for (int r = 1; r < numRows; r++) {
            rexp = expected[r];
            ract = actual[r];
            rchecked = checked[r];
            for (int c = 2; c < numCols; c++) {
                exp = rexp[c];
                act = ract[c];
                diff = act - exp;
                expNorm += exp * exp;
                diffNorm += diff * diff;
                rchecked[c] = true;
            }
        }

        rexp = expected[0];
        ract = actual[0];
        rchecked = checked[0];
        for (int c = 2; c < numCols; c++) {
            exp = rexp[c];
            act = ract[c];
            diff = act - exp;
            expNorm += exp * exp;
            diffNorm += diff * diff;
            rchecked[c] = true;
        }

        for (int r = 1; r < numRows / 2; r++) {
            exp = expected[r][0];
            act = actual[r][0];
            diff = act - exp;
            expNorm += exp * exp;
            diffNorm += diff * diff;
            checked[r][0] = true;

            exp = expected[r][1];
            act = actual[r][1];
            diff = act - exp;
            expNorm += exp * exp;
            diffNorm += diff * diff;
            checked[r][1] = true;

            exp = expected[numRows - r][numCols];
            act = actual[numRows - r][1];
            diff = act - exp;
            expNorm += exp * exp;
            diffNorm += diff * diff;
            checked[numRows - r][1] = true;

            exp = expected[numRows - r][numCols + 1];
            act = actual[numRows - r][0];
            diff = act - exp;
            expNorm += exp * exp;
            diffNorm += diff * diff;
            checked[numRows - r][0] = true;
        }

        exp = expected[0][0];
        act = actual[0][0];
        diff = act - exp;
        expNorm += exp * exp;
        diffNorm += diff * diff;
        checked[0][0] = true;

        exp = expected[0][numCols];
        act = actual[0][1];
        diff = act - exp;
        expNorm += exp * exp;
        diffNorm += diff * diff;
        checked[0][1] = true;

        exp = expected[numRows / 2][0];
        act = actual[numRows / 2][0];
        diff = act - exp;
        expNorm += exp * exp;
        diffNorm += diff * diff;
        checked[numRows / 2][0] = true;

        exp = expected[numRows / 2][numCols];
        act = actual[numRows / 2][1];
        diff = act - exp;
        expNorm += exp * exp;
        diffNorm += diff * diff;
        checked[numRows / 2][1] = true;

        expNorm = Math.sqrt(expNorm);
        diffNorm = Math.sqrt(diffNorm);
        for (int r = 0; r < numRows; r++) {
            for (int c = 0; c < numCols; c++) {
                Assert.assertTrue(String.format("[%d][%d]", r, c),
                        checked[r][c]);
            }
        }
        checkRelativeError(expNorm, diffNorm);
    }

    /**
     * A test of {@link DoubleFFT_2D#realForwardFull(double[])}.
     */
    @Test
    public void testRealForwardFull1dInput() {
        final double[] actual = new double[2 * numRows * numCols];
        final double[][] expected = new double[numRows][2 * numCols];
        for (int r = 0; r < numRows; r++) {
            final double[] rexp = expected[r];
            final int offset = r * numCols;
            for (int c = 0; c < numCols; c++) {
                actual[offset + c] = random.nextDouble();
                rexp[2 * c] = actual[offset + c];
                rexp[2 * c + 1] = 0.0;
            }
        }
        fft.realForwardFull(actual);
        complexForward(expected);
        double diffNorm = 0.;
        double expNorm = 0.;
        for (int r = 0; r < numRows; r++) {
            final double[] rexp = expected[r];
            final int offset = 2 * r * numCols;
            for (int c = 0; c < 2 * numCols; c++) {
                final double exp = rexp[c];
                final double act = actual[offset + c];
                final double diff = act - exp;
                expNorm += exp * exp;
                diffNorm += diff * diff;
            }
        }
        expNorm = Math.sqrt(expNorm);
        diffNorm = Math.sqrt(diffNorm);
        checkRelativeError(expNorm, diffNorm);
    }

    /**
     * A test of {@link DoubleFFT_2D#realForwardFull(double[][])}.
     */
    @Test
    public void testRealForwardFull2dInput() {
        final double[][] actual = new double[numRows][2 * numCols];
        final double[][] expected = new double[numRows][2 * numCols];
        for (int r = 0; r < numRows; r++) {
            final double[] rexp = expected[r];
            final double[] ract = actual[r];
            for (int c = 0; c < numCols; c++) {
                ract[c] = random.nextDouble();
                rexp[2 * c] = ract[c];
                rexp[2 * c + 1] = 0.;
            }
        }
        fft.realForwardFull(actual);
        complexForward(expected);
        double diffNorm = 0.;
        double expNorm = 0.;
        for (int r = 0; r < numRows; r++) {
            final double[] rexp = expected[r];
            final double[] ract = actual[r];
            for (int c = 0; c < 2 * numCols; c++) {
                final double exp = rexp[c];
                final double act = ract[c];
                final double diff = act - exp;
                expNorm += exp * exp;
                diffNorm += diff * diff;
            }
        }
        expNorm = Math.sqrt(expNorm);
        diffNorm = Math.sqrt(diffNorm);
        checkRelativeError(expNorm, diffNorm);
    }

    /**
     * A test of {@link DoubleFFT_2D#realInverseFull(double[], boolean)}, with
     * the second parameter set to <code>true</code>.
     */
    @Test
    public void testRealInverseFullScaled1dInput() {
        final double[] actual = new double[2 * numRows * numCols];
        final double[] expected = new double[2 * numRows * numCols];
        for (int r = 0; r < numRows; r++) {
            for (int c = 0; c < numCols; c++) {
                actual[r * numCols + c] = random.nextDouble();
                expected[2 * r * numCols + 2 * c] = actual[r * numCols + c];
                expected[2 * r * numCols + 2 * c + 1] = 0.0;
            }
        }
        fft.realInverseFull(actual, true);
        fft.complexInverse(expected, true);
        double diffNorm = 0.;
        double expNorm = 0.;
        for (int i = 0; i < actual.length; i++) {
            final double exp = expected[i];
            final double act = actual[i];
            final double diff = act - exp;
            expNorm += exp * exp;
            diffNorm += diff * diff;
        }
        expNorm = Math.sqrt(expNorm);
        diffNorm = Math.sqrt(diffNorm);
        checkRelativeError(expNorm, diffNorm);
    }

    /**
     * A test of {@link DoubleFFT_2D#realInverseFull(double[][], boolean)}, with
     * the second parameter set to <code>true</code>.
     */
    @Test
    public void testRealInverseFullScaled2dInput() {
        final double[][] actual = new double[numRows][2 * numCols];
        final double[][] expected = new double[numRows][2 * numCols];
        for (int r = 0; r < numRows; r++) {
            final double[] rexp = expected[r];
            final double[] ract = actual[r];
            for (int c = 0; c < numCols; c++) {
                ract[c] = random.nextDouble();
                rexp[2 * c] = ract[c];
                rexp[2 * c + 1] = 0.0;
            }
        }
        fft.realInverseFull(actual, true);
        fft.complexInverse(expected, true);

        double diffNorm = 0.;
        double expNorm = 0.;
        for (int r = 0; r < numRows; r++) {
            final double[] ract = actual[r];
            final double[] rexp = expected[r];
            for (int c = 0; c < 2 * numCols; c++) {
                final double exp = rexp[c];
                final double act = ract[c];
                final double diff = act - exp;
                expNorm += exp * exp;
                diffNorm += diff * diff;
            }
        }
        expNorm = Math.sqrt(expNorm);
        diffNorm = Math.sqrt(diffNorm);
        checkRelativeError(expNorm, diffNorm);
    }

    /**
     * A test of {@link DoubleFFT_2D#realInverseFull(double[], boolean)}, with
     * the second parameter set to <code>false</code>.
     */
    @Test
    public void testRealInverseFullUnscaled1dInput() {
        final double[] actual = new double[2 * numRows * numCols];
        final double[] expected = new double[2 * numRows * numCols];
        for (int r = 0; r < numRows; r++) {
            for (int c = 0; c < numCols; c++) {
                actual[r * numCols + c] = random.nextDouble();
                expected[2 * r * numCols + 2 * c] = actual[r * numCols + c];
                expected[2 * r * numCols + 2 * c + 1] = 0.0;
            }
        }
        fft.realInverseFull(actual, false);
        fft.complexInverse(expected, false);
        double diffNorm = 0.;
        double expNorm = 0.;
        for (int i = 0; i < actual.length; i++) {
            final double exp = expected[i];
            final double act = actual[i];
            final double diff = act - exp;
            expNorm += exp * exp;
            diffNorm += diff * diff;
        }
        expNorm = Math.sqrt(expNorm);
        diffNorm = Math.sqrt(diffNorm);
        checkRelativeError(expNorm, diffNorm);
    }

    /**
     * A test of {@link DoubleFFT_2D#realInverseFull(double[][], boolean)}, with
     * the second parameter set to <code>false</code>.
     */
    @Test
    public void testRealInverseFullUnscaled2dInput() {
        final double[][] actual = new double[numRows][2 * numCols];
        final double[][] expected = new double[numRows][2 * numCols];
        for (int r = 0; r < numRows; r++) {
            final double[] rexp = expected[r];
            final double[] ract = actual[r];
            for (int c = 0; c < numCols; c++) {
                ract[c] = random.nextDouble();
                rexp[2 * c] = ract[c];
                rexp[2 * c + 1] = 0.0;
            }
        }
        fft.realInverseFull(actual, false);
        fft.complexInverse(expected, false);

        double diffNorm = 0.;
        double expNorm = 0.;
        for (int r = 0; r < numRows; r++) {
            final double[] ract = actual[r];
            final double[] rexp = expected[r];
            for (int c = 0; c < 2 * numCols; c++) {
                final double exp = rexp[c];
                final double act = ract[c];
                final double diff = act - exp;
                expNorm += exp * exp;
                diffNorm += diff * diff;
            }
        }
        expNorm = Math.sqrt(expNorm);
        diffNorm = Math.sqrt(diffNorm);
        checkRelativeError(expNorm, diffNorm);
    }

    /**
     * A test of {@link DoubleFFT_2D#realInverse(double[], boolean)}, with the
     * second parameter set to <code>true</code>.
     */
    @Test
    public void testRealInverseScaled1dInput() {
        if ((!ConcurrencyUtils.isPowerOf2(numRows))
                || (!ConcurrencyUtils.isPowerOf2(numCols))) {
            return;
        }
        final double[] actual = new double[numRows * numCols];
        final double[] expected = new double[actual.length];
        for (int i = 0; i < actual.length; i++) {
            actual[i] = random.nextDouble();
            expected[i] = actual[i];
        }
        fft.realForward(actual);
        fft.realInverse(actual, true);
        double expNorm = 0.0;
        double diffNorm = 0.0;
        for (int i = 0; i < actual.length; i++) {
            final double exp = expected[i];
            final double act = actual[i];
            final double diff = act - exp;
            expNorm += exp * exp;
            diffNorm += diff * diff;
        }
        expNorm = Math.sqrt(expNorm);
        diffNorm = Math.sqrt(diffNorm);
        checkRelativeError(expNorm, diffNorm);
    }

    /**
     * A test of {@link DoubleFFT_2D#realInverse(double[][], boolean)}, with the
     * second parameter set to <code>true</code>.
     */
    @Test
    public void testRealInverseScaled2dInput() {
        if ((!ConcurrencyUtils.isPowerOf2(numRows))
                || (!ConcurrencyUtils.isPowerOf2(numCols))) {
            return;
        }
        final double[][] actual = new double[numRows][numCols];
        final double[][] expected = new double[numRows][numCols];
        for (int r = 0; r < numRows; r++) {
            final double[] rexp = expected[r];
            final double[] ract = actual[r];
            for (int c = 0; c < numCols; c++) {
                ract[c] = random.nextDouble();
                rexp[c] = ract[c];
            }
        }
        fft.realForward(actual);
        fft.realInverse(actual, true);
        double expNorm = 0.0;
        double diffNorm = 0.0;
        for (int r = 0; r < numRows; r++) {
            final double[] rexp = expected[r];
            final double[] ract = actual[r];
            for (int c = 0; c < numCols; c++) {
                final double exp = rexp[c];
                final double act = ract[c];
                final double diff = act - exp;
                expNorm += exp * exp;
                diffNorm += diff * diff;
            }
        }
        expNorm = Math.sqrt(expNorm);
        diffNorm = Math.sqrt(diffNorm);
        checkRelativeError(expNorm, diffNorm);
    }
}
