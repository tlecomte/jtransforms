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
 * The Original Code is Parallel Colt.
 *
 * The Initial Developer of the Original Code is
 * Piotr Wendykier, Emory University.
 * Portions created by the Initial Developer are Copyright (C) 2007
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
package edu.emory.mathcs.utils;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;

/**
 * Concurrency utilities.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class ConcurrencyUtils {
    /**
     * Thread pool.
     */
    public static ExecutorService threadPool = Executors.newCachedThreadPool(new CustomThreadFactory(new CustomExceptionHandler()));

    private static int THREADS_BEGIN_N_1D_FFT_2THREADS = 8192;

    private static int THREADS_BEGIN_N_1D_FFT_4THREADS = 65536;

    private static int THREADS_BEGIN_N_2D = 65536;

    private static int THREADS_BEGIN_N_3D = 65536;

    private static int np = concurrency();

    private ConcurrencyUtils() {

    }

    private static class CustomExceptionHandler implements Thread.UncaughtExceptionHandler {
        public void uncaughtException(Thread t, Throwable e) {
            e.printStackTrace();
        }

    }

    private static class CustomThreadFactory implements ThreadFactory {
        private static final ThreadFactory defaultFactory = Executors.defaultThreadFactory();

        private final Thread.UncaughtExceptionHandler handler;

        CustomThreadFactory(Thread.UncaughtExceptionHandler handler) {
            this.handler = handler;
        }

        public Thread newThread(Runnable r) {
            Thread t = defaultFactory.newThread(r);
            t.setUncaughtExceptionHandler(handler);
            return t;
        }
    };

    /**
     * Returns the number of available processors.
     * 
     * @return number of available processors
     */
    public static int concurrency() {
        int availableProcessors = Runtime.getRuntime().availableProcessors();
        if (availableProcessors > 1) {
            return prevPow2(availableProcessors);
        } else {
            return 1;
        }
    }

    /**
     * Returns the number of available processors ( = number of threads used in
     * calculations).
     * 
     * @return the number of available processors
     */
    public static int getNumberOfProcessors() {
        return np;
    }

    /**
     * Sets the number of available processors ( = number of threads used in
     * calculations). If n is not a power-of-two number, then the number of
     * available processors is set to the closest power-of-two number less than
     * n.
     * 
     * @param n
     * @return the number of available processors
     */
    public static int setNumberOfProcessors(int n) {
        if (isPowerOf2(n)) {
            np = n;
        } else {
            np = prevPow2(n);
        }
        // if (n > 0) {
        // np = n;
        // }
        // else {
        // np = concurrency();
        // }
        return np;
    }

    /**
     * Returns the minimal size of 1D data for which two threads are used.
     * 
     * @return the minimal size of 1D data for which two threads are used
     */
    public static int getThreadsBeginN_1D_FFT_2Threads() {
        return THREADS_BEGIN_N_1D_FFT_2THREADS;
    }

    /**
     * Returns the minimal size of 1D data for which four threads are used.
     * 
     * @return the minimal size of 1D data for which four threads are used
     */
    public static int getThreadsBeginN_1D_FFT_4Threads() {
        return THREADS_BEGIN_N_1D_FFT_4THREADS;
    }

    /**
     * Returns the minimal size of 2D data for which threads are used.
     * 
     * @return the minimal size of 2D data for which threads are used
     */
    public static int getThreadsBeginN_2D() {
        return THREADS_BEGIN_N_2D;
    }

    /**
     * Returns the minimal size of 3D data for which threads are used.
     * 
     * @return the minimal size of 3D data for which threads are used
     */
    public static int getThreadsBeginN_3D() {
        return THREADS_BEGIN_N_3D;
    }

    /**
     * Sets the minimal size of 1D data for which two threads are used.
     * 
     * @param n
     *            the minimal size of 1D data for which two threads are used
     */
    public static void setThreadsBeginN_1D_FFT_2Threads(int n) {
        if (n < 512) {
            THREADS_BEGIN_N_1D_FFT_2THREADS = 512;
        } else {
            THREADS_BEGIN_N_1D_FFT_2THREADS = n;
        }
    }

    /**
     * Sets the minimal size of 1D data for which four threads are used.
     * 
     * @param n
     *            the minimal size of 1D data for which four threads are used
     */
    public static void setThreadsBeginN_1D_FFT_4Threads(int n) {
        if (n < 512) {
            THREADS_BEGIN_N_1D_FFT_4THREADS = 512;
        } else {
            THREADS_BEGIN_N_1D_FFT_4THREADS = n;
        }
    }

    /**
     * Sets the minimal size of 2D data for which threads are used.
     * 
     * @param n
     *            the minimal size of 2D data for which threads are used
     */
    public static void setThreadsBeginN_2D(int n) {
        THREADS_BEGIN_N_2D = n;
    }

    /**
     * Sets the minimal size of 3D data for which threads are used.
     * 
     * @param n
     *            the minimal size of 3D data for which threads are used
     */
    public static void setThreadsBeginN_3D(int n) {
        THREADS_BEGIN_N_3D = n;
    }

    /**
     * Resets the minimal size of 1D data for which two and four threads are
     * used.
     */
    public static void resetThreadsBeginN_FFT() {
        THREADS_BEGIN_N_1D_FFT_2THREADS = 8192;
        THREADS_BEGIN_N_1D_FFT_4THREADS = 65536;
    }

    /**
     * Resets the minimal size of 2D and 3D data for which threads are used.
     */
    public static void resetThreadsBeginN() {
        THREADS_BEGIN_N_2D = 65536;
        THREADS_BEGIN_N_3D = 65536;
    }

    /**
     * Returns the closest power-of-two number greater than or equal to x.
     * 
     * @param x
     * @return the closest power-of-two number greater than or equal to x
     */
    public static int nextPow2(int x) {
        if (x < 1)
            throw new IllegalArgumentException("x must be greater or equal 1");
        x |= (x >>> 1);
        x |= (x >>> 2);
        x |= (x >>> 4);
        x |= (x >>> 8);
        x |= (x >>> 16);
        x |= (x >>> 32);
        return x + 1;
    }

    /**
     * Returns the closest power-of-two number less than or equal to x.
     * 
     * @param x
     * @return the closest power-of-two number less then or equal to x
     */
    public static int prevPow2(int x) {
        if (x < 1)
            throw new IllegalArgumentException("x must be greater or equal 1");
        return (int) Math.pow(2, Math.floor(Math.log(x) / Math.log(2)));
    }

    /**
     * Checks if x is a power-of-two number.
     * 
     * @param x
     * @return true if x is a power-of-two number
     */
    public static boolean isPowerOf2(int x) {
        if (x <= 0)
            return false;
        else
            return (x & (x - 1)) == 0;
    }
}
