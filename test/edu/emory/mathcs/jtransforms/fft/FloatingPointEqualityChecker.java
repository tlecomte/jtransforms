package edu.emory.mathcs.jtransforms.fft;

import org.junit.Assert;

/**
 * A utility class for consistently asserting equality of two floating-point
 * numbers.
 *
 * @author S&eacute;bastien Brisard
 *
 */
public class FloatingPointEqualityChecker {
    /**
     * If the expected (<code>double</code>) value is below this threshold, the
     * relative error is not tested.
     */
    private final double dabs;

    /** Maximum (<code>double</code> relative error. */
    private final double drel;

    /**
     * If the expected (<code>double</code>) value is below this threshold, the
     * relative error is not tested.
     */
    private final float fabs;

    /** Maximum (<code>float</code> relative error. */
    private final double frel;

    /** Default message used in all thrown exceptions. */
    private final String msg;

    /**
     * Creates a new instance of this class.
     *
     * @param msg
     *            the default message returned by all assertion exceptions
     * @param drel
     *            the maximum relative error, for <code>double</code> values
     * @param dabs
     *            the maximum absolute error, for <code>double</code> values
     * @param frel
     *            the maximum relative error, for <code>float</code> values
     * @param fabs
     *            the maximum absolute error, for <code>float</code> values
     */
    public FloatingPointEqualityChecker(final String msg, final double drel,
            final double dabs, final float frel, final float fabs) {
        this.msg = msg;
        this.drel = drel;
        this.dabs = dabs;
        this.frel = frel;
        this.fabs = fabs;
    }

    /**
     * Asserts that two <code>double</code>s are equal.
     *
     * @param msg
     *            a message to be concatenated with the default message
     * @param expected
     *            expected value
     * @param actual
     *            the value to check against <code>expected</code>
     */
    public final void assertEquals(final String msg, final double expected,
            final double actual) {
        final double delta = Math.abs(actual - expected);
        if (!(delta <= drel * Math.abs(expected))) {
            Assert.assertEquals(this.msg + msg + ", abs = " + delta
                    + ", rel = " + (delta / Math.abs(expected)), expected,
                    actual, dabs);
        }
    }

    /**
     * Asserts that two <code>float</code>s are equal.
     *
     * @param msg
     *            a message to be concatenated with the default message
     * @param expected
     *            expected value
     * @param actual
     *            the value to check against <code>expected</code>
     */
    public final void assertEquals(final String msg, final float expected,
            final float actual) {
        final float delta = Math.abs(actual - expected);
        if (!(delta <= frel * Math.abs(expected))) {
            Assert.assertEquals(this.msg + msg + ", abs = " + delta
                    + ", rel = " + (delta / Math.abs(expected)), expected,
                    actual, fabs);
        }
    }
}
