package com.google.mlkit.vision.demo.tflite

import android.app.Activity
import org.tensorflow.lite.support.common.TensorOperator
import org.tensorflow.lite.support.common.ops.NormalizeOp

class AgeClassifier
/**
 * Initializes a `ClassifierFloatMobileNet`.
 *
 * @param activity
 */
    (activity: Activity) : Classifier(activity) {
    override val modelPath: String
        get() =// you can download this file from
        // see build.gradle for where to obtain this file. It should be auto
            // downloaded into assets.
            "age224.tflite"

    override val labelPath: String
        get() = "age_label.txt"

    override val preprocessNormalizeOp: TensorOperator
        get() = NormalizeOp(IMAGE_MEAN, IMAGE_STD)

    override val postprocessNormalizeOp: TensorOperator
        get() = NormalizeOp(PROBABILITY_MEAN, PROBABILITY_STD)

    companion object {
        /** Float MobileNet requires additional normalization of the used input.  */
        private const val IMAGE_MEAN = 0f

        private const val IMAGE_STD = 255f

        /**
         * Float model does not need dequantization in the post-processing. Setting mean and std as 0.0f
         * and 1.0f, repectively, to bypass the normalization.
         */
        private const val PROBABILITY_MEAN = 0.0f

        private const val PROBABILITY_STD = 1.0f
    }
}

