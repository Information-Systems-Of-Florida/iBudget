USE [APD]
GO

-- Add Claims field if it doesn't exist
IF NOT EXISTS (
    SELECT * 
    FROM sys.columns 
    WHERE object_id = OBJECT_ID(N'[dbo].[tbl_Model5b_Predictions]') 
    AND name = 'Claims'
)
BEGIN
    ALTER TABLE [dbo].[tbl_Model5b_Predictions]
    ADD Claims float NULL;
    
    PRINT 'Added Claims column to tbl_Model5b_Predictions';
END
ELSE
BEGIN
    PRINT 'Claims column already exists in tbl_Model5b_Predictions';
END
GO

-- Add Prediction field (squared value) if it doesn't exist
IF NOT EXISTS (
    SELECT * 
    FROM sys.columns 
    WHERE object_id = OBJECT_ID(N'[dbo].[tbl_Model5b_Predictions]') 
    AND name = 'Prediction'
)
BEGIN
    ALTER TABLE [dbo].[tbl_Model5b_Predictions]
    ADD Prediction float NULL;
    
    PRINT 'Added Prediction column to tbl_Model5b_Predictions';
END
ELSE
BEGIN
    PRINT 'Prediction column already exists in tbl_Model5b_Predictions';
END
GO

-- Add ModelCoefficients field to store JSON coefficients if it doesn't exist
IF NOT EXISTS (
    SELECT * 
    FROM sys.columns 
    WHERE object_id = OBJECT_ID(N'[dbo].[tbl_Model5b_Predictions]') 
    AND name = 'ModelCoefficients'
)
BEGIN
    ALTER TABLE [dbo].[tbl_Model5b_Predictions]
    ADD ModelCoefficients nvarchar(max) NULL;
    
    PRINT 'Added ModelCoefficients column to tbl_Model5b_Predictions';
END
ELSE
BEGIN
    PRINT 'ModelCoefficients column already exists in tbl_Model5b_Predictions';
END
GO

-- Add CalibrationDate field if it doesn't exist
IF NOT EXISTS (
    SELECT * 
    FROM sys.columns 
    WHERE object_id = OBJECT_ID(N'[dbo].[tbl_Model5b_Predictions]') 
    AND name = 'CalibrationDate'
)
BEGIN
    ALTER TABLE [dbo].[tbl_Model5b_Predictions]
    ADD CalibrationDate datetime NULL;
    
    PRINT 'Added CalibrationDate column to tbl_Model5b_Predictions';
END
ELSE
BEGIN
    -- Update column name if it exists but named differently
    PRINT 'CalibrationDate column already exists or CalculationDate is being used';
END
GO

-- Add ModelVersion field to track which version of the model was used
IF NOT EXISTS (
    SELECT * 
    FROM sys.columns 
    WHERE object_id = OBJECT_ID(N'[dbo].[tbl_Model5b_Predictions]') 
    AND name = 'ModelVersion'
)
BEGIN
    ALTER TABLE [dbo].[tbl_Model5b_Predictions]
    ADD ModelVersion varchar(50) NULL;
    
    PRINT 'Added ModelVersion column to tbl_Model5b_Predictions';
END
ELSE
BEGIN
    PRINT 'ModelVersion column already exists in tbl_Model5b_Predictions';
END
GO

-- Create stored procedure to update claims amounts for a specific date range
IF EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[sp_Update_Model5b_Claims]') AND type in (N'P', N'PC'))
DROP PROCEDURE [dbo].[sp_Update_Model5b_Claims]
GO

CREATE PROCEDURE [dbo].[sp_Update_Model5b_Claims]
    @StartDate datetime = '2024-07-01',
    @EndDate datetime = '2025-06-30',
    @RunID bigint = NULL
AS
BEGIN
    SET NOCOUNT ON;
    
    -- Update Claims amount for existing predictions
    UPDATE p
    SET p.Claims = c.TotalClaims
    FROM [dbo].[tbl_Model5b_Predictions] p
    INNER JOIN (
        SELECT 
            CaseNo,
            SUM(PaidAmt) AS TotalClaims
        FROM [dbo].[tbl_Claims_MMIS]
        WHERE ServiceDate >= @StartDate 
        AND ServiceDate < @EndDate
        GROUP BY CaseNo
    ) c ON p.CaseNo = c.CaseNo
    WHERE (@RunID IS NULL OR p.RunID = @RunID);
    
    -- Return count of updated records
    SELECT @@ROWCOUNT AS RecordsUpdated;
    
END
GO

GRANT EXECUTE ON [dbo].[sp_Update_Model5b_Claims] TO [public]
GO