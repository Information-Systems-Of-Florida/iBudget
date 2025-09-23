-- =============================================
-- Author:      APD Data Team
-- Create date: 2025-09-19
-- Description: Implements Model 5b for Florida APD iBudget Algorithm
--              Based on UpdateStatisticalModelsiBudget document
--              This procedure pulls QSI assessment data and calculates 
--              predicted budget allocations using square-root transformation
-- =============================================


-- =============================================
-- SUPPORTING TABLE FOR STORING PREDICTIONS: -- drop table dbo.tbl_Model5b_Predictions
-- =============================================
CREATE TABLE dbo.tbl_Model5b_Predictions (
    PredictionID BIGINT IDENTITY(1,1) PRIMARY KEY,
    CaseNo BIGINT NOT NULL,
    AssessID BIGINT NOT NULL,
    ReviewDate DATETIME,
    LivingSetting VARCHAR(10),
    Age INT,
    AgeGroup VARCHAR(20),
    BSum INT,
    FSum INT,
    PSum INT,
    FHFSum INT,
    SLFSum INT,
    SLBSum INT,
    SqrtPrediction DECIMAL(19, 4),
    PredictedBudget DECIMAL(19, 2),
    CalculationDate DATETIME,
    CreatedDate DATETIME DEFAULT GETDATE(),
    CreatedBy VARCHAR(100),
    RunID BIGINT NOT NULL,
    INDEX IX_RunID (RunID),
    INDEX IX_CaseNo (CaseNo),
    INDEX IX_AssessID (AssessID),
    INDEX IX_CalculationDate (CalculationDate)
);

-- =============================================
-- HELPER FUNCTION: Get Living Setting from Services -- drop FUNCTION dbo.fn_GetLivingSetting
-- This would need to be customized based on actual business rules
-- =============================================
CREATE FUNCTION dbo.fn_GetLivingSetting(@CaseNo BIGINT)
RETURNS VARCHAR(10)
AS
BEGIN
    DECLARE @LivingSetting VARCHAR(10);
    
    -- Complex logic to determine living setting from services
    -- This is a simplified version - actual implementation would need
    -- proper mapping based on APD business rules and service codes
    
    -- First check RESIDENCETYPE in Consumers table
    SELECT @LivingSetting = 
        CASE 
            WHEN RESIDENCETYPE LIKE '%RH4%' OR RESIDENCETYPE LIKE '%CTEP%' THEN 'RH4'
            WHEN RESIDENCETYPE LIKE '%RH3%' OR RESIDENCETYPE LIKE '%Intensive%' THEN 'RH3'
            WHEN RESIDENCETYPE LIKE '%RH2%' OR RESIDENCETYPE LIKE '%Behavior Focus%' THEN 'RH2'
            WHEN RESIDENCETYPE LIKE '%RH1%' OR RESIDENCETYPE LIKE '%Residential%' THEN 'RH1'
            WHEN RESIDENCETYPE LIKE '%Independent%' OR RESIDENCETYPE LIKE '%Supported%' THEN 'ILSL'
            ELSE NULL
        END
    FROM dbo.tbl_Consumers
    WHERE CASENO = @CaseNo;
    
    -- If not determined from RESIDENCETYPE, check services
    IF @LivingSetting IS NULL
    BEGIN
        SELECT TOP 1 @LivingSetting = 
            CASE 
                WHEN ServiceCode LIKE 'RH%' THEN 
                    CASE 
                        WHEN ServiceCode IN ('RHCTEP1', 'RHCTEP2', 'RHCTEP3', 'RHCTEP4', 'SHC') THEN 'RH4'
                        WHEN ServiceCode IN ('RHIB1', 'RHIB2', 'RHIB3', 'RHIB4') THEN 'RH3'
                        WHEN ServiceCode IN ('RHBF1', 'RHBF2', 'RHBF3', 'RHBF4') THEN 'RH2'
                        ELSE 'RH1'
                    END
                WHEN ServiceCode IN ('ILS', 'SLS', 'ILSL') THEN 'ILSL'
                ELSE 'FH'
            END
        FROM dbo.tbl_PlannedServices
        WHERE CaseNo = @CaseNo
            AND PlannedServiceStatus = 'Active'
        ORDER BY 
            CASE 
                WHEN ServiceCode LIKE 'RH%' THEN 1
                WHEN ServiceCode IN ('ILS', 'SLS', 'ILSL') THEN 2
                ELSE 3
            END;
    END
    
    RETURN ISNULL(@LivingSetting, 'FH');
END
