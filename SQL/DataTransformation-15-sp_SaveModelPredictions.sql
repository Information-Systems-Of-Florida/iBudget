-- Save model predictions
CREATE OR ALTER PROCEDURE sp_SaveModelPredictions
    @ModelID INT,
    @ModelName VARCHAR(100),
    @ConsumerID INT,
    @PredictedBudget DECIMAL(18,2),
    @FiscalYear VARCHAR(10)
AS
BEGIN
    INSERT INTO tbl_ModelPredictions (
        ModelID, ModelName, ConsumerID, 
        PredictedBudget, FiscalYear, PredictionDate
    )
    VALUES (
        @ModelID, @ModelName, @ConsumerID,
        @PredictedBudget, @FiscalYear, GETDATE()
    )
END
GO

-- Save model metrics
CREATE OR ALTER PROCEDURE sp_SaveModelMetrics
    @ModelID INT,
    @ModelName VARCHAR(100),
    @MetricType VARCHAR(50),
    @MetricValue DECIMAL(18,6),
    @DataSet VARCHAR(20)  -- 'train', 'test', 'cv'
AS
BEGIN
    INSERT INTO tbl_ModelMetrics (
        ModelID, ModelName, MetricType, 
        MetricValue, DataSet, Timestamp
    )
    VALUES (
        @ModelID, @ModelName, @MetricType,
        @MetricValue, @DataSet, GETDATE()
    )
END
GO