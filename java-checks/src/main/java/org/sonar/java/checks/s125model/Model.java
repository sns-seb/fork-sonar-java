package org.sonar.java.checks.s125model;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Iterator;
import java.util.stream.Stream;

import static java.nio.charset.StandardCharsets.UTF_8;

public class Model {
    private final ModelParams modelParams;
    private final double threshold;

    public Model(ModelParams modelParams, double threshold) {
        this.modelParams = modelParams;
        this.threshold = threshold;
    }

    public static Model create(InputStream modelFile, double decisionThreshold) {
        ModelParams modelParams = readModelParams(modelFile);
        Model model = new Model(modelParams, decisionThreshold);
//        System.out.println("Model:");
//        System.out.println(Arrays.toString(modelParams.getCoefficients()));
//        System.out.println(modelParams.getIntercept());
        return model;
    }

    private static ModelParams readModelParams(InputStream modelFile) {
        JsonObject asJsonObject = JsonParser.parseReader(new InputStreamReader(modelFile, UTF_8)).getAsJsonObject();
        Iterator<JsonElement> it = asJsonObject.getAsJsonArray("coefficients").iterator();
        double intercept = asJsonObject.get("intercept").getAsDouble();
        double[] coefficients = Stream.generate(() -> null)
                .takeWhile(x -> it.hasNext())
                .map(n -> it.next())
                .mapToDouble(JsonElement::getAsDouble)
                .toArray();
        return new ModelParams(intercept, coefficients);
    }

    public Prediction predict(double[] features) {
        double linearRegression = linearRegression(features);
        double normalizedRegression = Model.normalize(linearRegression);
        int decision = decide(normalizedRegression);
        return new Prediction(linearRegression, normalizedRegression, decision);
    }

    private double linearRegression(double[] features) {
        double[] coefficients = modelParams.getCoefficients();

        double raw = modelParams.getIntercept();
        for (int i = 0; i < coefficients.length; i++) {
            raw += features[i] * coefficients[i];
        }
        return raw;
    }

    private static double normalize(double raw) {
        return 1 / (1 + Math.exp(-raw));
    }

    private int decide(double sig) {
        if (sig > threshold) {
            return 1;
        }
        return 0;
    }

    public static class Prediction {
        private final double linearRegression;
        private final double normalizedRegression;
        private final int decision;

        Prediction(double linearRegression, double normalizedRegression, int decision) {
            this.linearRegression = linearRegression;
            this.normalizedRegression = normalizedRegression;
            this.decision = decision;
        }

        public double getLinearRegression() {
            return linearRegression;
        }

        public double getNormalizedRegression() {
            return normalizedRegression;
        }

        public int getDecision() {
            return decision;
        }
    }

    static class ModelParams {
        private final double intercept;
        private final double[] coefficients;

        public ModelParams(double intercept, double[] coefficients) {
            this.intercept = intercept;
            this.coefficients = coefficients;
        }

        public double getIntercept() {
            return intercept;
        }

        public double[] getCoefficients() {
            return coefficients;
        }
    }
}
