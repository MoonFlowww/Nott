#ifndef Nott_METRIC_HPP
#define Nott_METRIC_HPP

namespace Nott::Metric::Classification {
    enum class Kind {
        Accuracy,
        AUCROC,
        BalancedAccuracy,
        BalancedErrorRate,
        F1,
        FBeta0Point5,
        FBeta2,
        FalseDiscoveryRate,
        FalseNegativeRate,
        FalseOmissionRate,
        FalsePositiveRate,
        FowlkesMallows,
        HammingLoss,
        Informedness,
        JaccardIndexMicro,
        JaccardIndexMacro,
        Markness,
        Matthews,
        NegativeLikelihoodRatio,
        NegativePredictiveValue,
        PositiveLikelihoodRatio,
        PositivePredictiveValue,
        Precision,
        Prevalence,
        Recall,
        Top1Error,
        Top3Error,
        Top5Error,
        Top1Accuracy,
        Top3Accuracy,
        Top5Accuracy,
        Specificity,
        ThreatScore,
        TrueNegativeRate,
        TruePositiveRate,
        YoudenIndex,
        LogLoss,
        BrierScore,
        BrierSkillScore,
        ExpectedCalibrationError,
        MaximumCalibrationError,
        CalibrationSlope,
        CalibrationIntercept,
        HosmerLemeshowPValue,
        KolmogorovSmirnovStatistic,
        CohensKappa,
        ConfusionEntropy,
        CoverageError,
        LabelRankingAveragePrecision,
        SubsetAccuracy,
        AUPRC,
        AUPRG,
        GiniCoefficient,
        HausdorffDistance,
        BoundaryIoU,
    };

    struct Descriptor {
        Kind kind;
    };

    [[nodiscard]] constexpr auto Make(Kind kind) noexcept -> Descriptor { return Descriptor{kind}; }

    inline constexpr Descriptor Accuracy{Kind::Accuracy};
    inline constexpr Descriptor AUCROC{Kind::AUCROC};
    inline constexpr Descriptor BalancedAccuracy{Kind::BalancedAccuracy};
    inline constexpr Descriptor BalancedErrorRate{Kind::BalancedErrorRate};
    inline constexpr Descriptor F1{Kind::F1};
    inline constexpr Descriptor FBeta0Point5{Kind::FBeta0Point5};
    inline constexpr Descriptor FBeta2{Kind::FBeta2};
    inline constexpr Descriptor FalseDiscoveryRate{Kind::FalseDiscoveryRate};
    inline constexpr Descriptor FalseNegativeRate{Kind::FalseNegativeRate};
    inline constexpr Descriptor FalseOmissionRate{Kind::FalseOmissionRate};
    inline constexpr Descriptor FalsePositiveRate{Kind::FalsePositiveRate};
    inline constexpr Descriptor FowlkesMallows{Kind::FowlkesMallows};
    inline constexpr Descriptor HammingLoss{Kind::HammingLoss};
    inline constexpr Descriptor Informedness{Kind::Informedness};
    inline constexpr Descriptor JaccardIndexMicro{Kind::JaccardIndexMicro};
    inline constexpr Descriptor JaccardIndexMacro{Kind::JaccardIndexMacro};
    inline constexpr Descriptor Markness{Kind::Markness};
    inline constexpr Descriptor Matthews{Kind::Matthews};
    inline constexpr Descriptor NegativeLikelihoodRatio{Kind::NegativeLikelihoodRatio};
    inline constexpr Descriptor NegativePredictiveValue{Kind::NegativePredictiveValue};
    inline constexpr Descriptor PositiveLikelihoodRatio{Kind::PositiveLikelihoodRatio};
    inline constexpr Descriptor PositivePredictiveValue{Kind::PositivePredictiveValue};
    inline constexpr Descriptor Precision{Kind::Precision};
    inline constexpr Descriptor Prevalence{Kind::Prevalence};
    inline constexpr Descriptor Recall{Kind::Recall};
    inline constexpr Descriptor Top1Error{Kind::Top1Error};
    inline constexpr Descriptor Top3Error{Kind::Top3Error};
    inline constexpr Descriptor Top5Error{Kind::Top5Error};
    inline constexpr Descriptor Top1Accuracy{Kind::Top1Accuracy};
    inline constexpr Descriptor Top3Accuracy{Kind::Top3Accuracy};
    inline constexpr Descriptor Top5Accuracy{Kind::Top5Accuracy};
    inline constexpr Descriptor Specificity{Kind::Specificity};
    inline constexpr Descriptor ThreatScore{Kind::ThreatScore};
    inline constexpr Descriptor TrueNegativeRate{Kind::TrueNegativeRate};
    inline constexpr Descriptor TruePositiveRate{Kind::TruePositiveRate};
    inline constexpr Descriptor YoudenIndex{Kind::YoudenIndex};
    inline constexpr Descriptor LogLoss{Kind::LogLoss};
    inline constexpr Descriptor BrierScore{Kind::BrierScore};
    inline constexpr Descriptor BrierSkillScore{Kind::BrierSkillScore};
    inline constexpr Descriptor ExpectedCalibrationError{Kind::ExpectedCalibrationError};
    inline constexpr Descriptor MaximumCalibrationError{Kind::MaximumCalibrationError};
    inline constexpr Descriptor CalibrationSlope{Kind::CalibrationSlope};
    inline constexpr Descriptor CalibrationIntercept{Kind::CalibrationIntercept};
    inline constexpr Descriptor HosmerLemeshowPValue{Kind::HosmerLemeshowPValue};
    inline constexpr Descriptor KolmogorovSmirnovStatistic{Kind::KolmogorovSmirnovStatistic};
    inline constexpr Descriptor CohensKappa{Kind::CohensKappa};
    inline constexpr Descriptor ConfusionEntropy{Kind::ConfusionEntropy};
    inline constexpr Descriptor CoverageError{Kind::CoverageError};
    inline constexpr Descriptor LabelRankingAveragePrecision{Kind::LabelRankingAveragePrecision};
    inline constexpr Descriptor SubsetAccuracy{Kind::SubsetAccuracy};
    inline constexpr Descriptor AUPRC{Kind::AUPRC};
    inline constexpr Descriptor AUPRG{Kind::AUPRG};
    inline constexpr Descriptor GiniCoefficient{Kind::GiniCoefficient};
    inline constexpr Descriptor HausdorffDistance{Kind::HausdorffDistance};
    inline constexpr Descriptor BoundaryIoU{Kind::BoundaryIoU};
}

namespace Nott::Metric::Timeseries {
    enum class Kind {
        MeanAbsoluteError,
        MeanAbsolutePercentageError,
        MeanBiasError,
        MeanSquaredError,
        MedianAbsoluteError,
        R2Score,
        RootMeanSquaredError,
        SymmetricMeanAbsolutePercentageError,
        WeightedAbsolutePercentageError,
        MeanPercentageError,
        ExplainedVariance,
        TheilsU1,
        TheilsU2,
        MeanAbsoluteScaledError,
        RootMeanSquaredScaledError,
        MedianRelativeAbsoluteError,
        GeometricMeanRelativeAbsoluteError,
        OverallWeightedAverage,
        DynamicTimeWarpingDistance,
        TimeWarpEditDistance,
        SpectralDistance,
        CosineSimilarity,
        NegativeLogLikelihood,
        ContinuousRankedProbabilityScore,
        EnergyScore,
        PinballLossAverage,
        BrierScore,
        PredictionIntervalCoverageProbability,
        MeanPredictionIntervalWidth,
        WinklerScore,
        ConditionalCoverageError,
        QuantileCrossingRate,
        AutocorrelationOfResiduals,
        PartialAutocorrelationOfResiduals,
        LjungBoxStatistic,
        BoxPierceStatistic,
        DurbinWatsonStatistic,
        JarqueBeraStatistic,
        AndersonDarlingStatistic,
        BreuschPaganStatistic,
        WhiteStatistic,
        PopulationStabilityIndex,
        KullbackLeiblerDivergence,
        JensenShannonDivergence,
        WassersteinDistance,
        MaximumMeanDiscrepancy,
        LossDriftSlope,
        LossCusumStatistic,
        ResidualChangePointScore,
        QLIKE,
        LogVarianceMeanSquaredError,
        SqrtVarianceMeanSquaredError,
        msIC,
        msIR,
    };

    struct Descriptor {
        Kind kind;
    };

    [[nodiscard]] constexpr auto Make(Kind kind) noexcept -> Descriptor { return Descriptor{kind}; }
    inline constexpr Descriptor MeanAbsoluteError{Kind::MeanAbsoluteError};
    inline constexpr Descriptor MeanAbsolutePercentageError{Kind::MeanAbsolutePercentageError};
    inline constexpr Descriptor MeanBiasError{Kind::MeanBiasError};
    inline constexpr Descriptor MeanSquaredError{Kind::MeanSquaredError};
    inline constexpr Descriptor MedianAbsoluteError{Kind::MedianAbsoluteError};
    inline constexpr Descriptor R2Score{Kind::R2Score};
    inline constexpr Descriptor RootMeanSquaredError{Kind::RootMeanSquaredError};
    inline constexpr Descriptor SymmetricMeanAbsolutePercentageError{Kind::SymmetricMeanAbsolutePercentageError};
    inline constexpr Descriptor WeightedAbsolutePercentageError{Kind::WeightedAbsolutePercentageError};
    inline constexpr Descriptor MeanPercentageError{Kind::MeanPercentageError};
    inline constexpr Descriptor ExplainedVariance{Kind::ExplainedVariance};
    inline constexpr Descriptor TheilsU1{Kind::TheilsU1};
    inline constexpr Descriptor TheilsU2{Kind::TheilsU2};
    inline constexpr Descriptor MeanAbsoluteScaledError{Kind::MeanAbsoluteScaledError};
    inline constexpr Descriptor RootMeanSquaredScaledError{Kind::RootMeanSquaredScaledError};
    inline constexpr Descriptor MedianRelativeAbsoluteError{Kind::MedianRelativeAbsoluteError};
    inline constexpr Descriptor GeometricMeanRelativeAbsoluteError{Kind::GeometricMeanRelativeAbsoluteError};
    inline constexpr Descriptor OverallWeightedAverage{Kind::OverallWeightedAverage};
    inline constexpr Descriptor DynamicTimeWarpingDistance{Kind::DynamicTimeWarpingDistance};
    inline constexpr Descriptor TimeWarpEditDistance{Kind::TimeWarpEditDistance};
    inline constexpr Descriptor SpectralDistance{Kind::SpectralDistance};
    inline constexpr Descriptor CosineSimilarity{Kind::CosineSimilarity};
    inline constexpr Descriptor NegativeLogLikelihood{Kind::NegativeLogLikelihood};
    inline constexpr Descriptor ContinuousRankedProbabilityScore{Kind::ContinuousRankedProbabilityScore};
    inline constexpr Descriptor EnergyScore{Kind::EnergyScore};
    inline constexpr Descriptor PinballLossAverage{Kind::PinballLossAverage};
    inline constexpr Descriptor BrierScore{Kind::BrierScore};
    inline constexpr Descriptor PredictionIntervalCoverageProbability{Kind::PredictionIntervalCoverageProbability};
    inline constexpr Descriptor MeanPredictionIntervalWidth{Kind::MeanPredictionIntervalWidth};
    inline constexpr Descriptor WinklerScore{Kind::WinklerScore};
    inline constexpr Descriptor ConditionalCoverageError{Kind::ConditionalCoverageError};
    inline constexpr Descriptor QuantileCrossingRate{Kind::QuantileCrossingRate};
    inline constexpr Descriptor AutocorrelationOfResiduals{Kind::AutocorrelationOfResiduals};
    inline constexpr Descriptor PartialAutocorrelationOfResiduals{Kind::PartialAutocorrelationOfResiduals};
    inline constexpr Descriptor LjungBoxStatistic{Kind::LjungBoxStatistic};
    inline constexpr Descriptor BoxPierceStatistic{Kind::BoxPierceStatistic};
    inline constexpr Descriptor DurbinWatsonStatistic{Kind::DurbinWatsonStatistic};
    inline constexpr Descriptor JarqueBeraStatistic{Kind::JarqueBeraStatistic};
    inline constexpr Descriptor AndersonDarlingStatistic{Kind::AndersonDarlingStatistic};
    inline constexpr Descriptor BreuschPaganStatistic{Kind::BreuschPaganStatistic};
    inline constexpr Descriptor WhiteStatistic{Kind::WhiteStatistic};
    inline constexpr Descriptor PopulationStabilityIndex{Kind::PopulationStabilityIndex};
    inline constexpr Descriptor KullbackLeiblerDivergence{Kind::KullbackLeiblerDivergence};
    inline constexpr Descriptor JensenShannonDivergence{Kind::JensenShannonDivergence};
    inline constexpr Descriptor WassersteinDistance{Kind::WassersteinDistance};
    inline constexpr Descriptor MaximumMeanDiscrepancy{Kind::MaximumMeanDiscrepancy};
    inline constexpr Descriptor LossDriftSlope{Kind::LossDriftSlope};
    inline constexpr Descriptor LossCusumStatistic{Kind::LossCusumStatistic};
    inline constexpr Descriptor ResidualChangePointScore{Kind::ResidualChangePointScore};
    inline constexpr Descriptor QLIKE{Kind::QLIKE};
    inline constexpr Descriptor LogVarianceMeanSquaredError{Kind::LogVarianceMeanSquaredError};
    inline constexpr Descriptor SqrtVarianceMeanSquaredError{Kind::SqrtVarianceMeanSquaredError};
    inline constexpr Descriptor msIC{Kind::msIC};
    inline constexpr Descriptor msIR{Kind::msIR};
}


#endif //Nott_METRIC_HPP