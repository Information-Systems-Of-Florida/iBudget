-- =============================================
-- Author:      APD Data Team
-- Create date: 2025-09-19
-- Description: Implements Model 5b for Florida APD iBudget Algorithm
--              Based on UpdateStatisticalModelsiBudget document
--              This procedure pulls QSI assessment data and calculates 
--              predicted budget allocations using square-root transformation
-- =============================================
USE [APD]
GO

IF EXISTS (SELECT * FROM sys.objects WHERE type = 'P' AND name = 'sp_Calculate_Model5b_iBudget')
    DROP PROCEDURE sp_Calculate_Model5b_iBudget
GO

CREATE PROCEDURE [dbo].[sp_Calculate_Model5b_iBudget]
    @CaseNo BIGINT = NULL,  -- Optional: Calculate for specific case
    @BatchSize INT = 1000    -- Optional: Process in batches for large datasets
AS
-- =============================================
-- USAGE EXAMPLES:
-- =============================================
-- Calculate for all consumers:
-- EXEC sp_Calculate_Model5b_iBudget;

-- Calculate for specific consumer:
-- EXEC sp_Calculate_Model5b_iBudget @CaseNo = 12345;

-- Process in batches:
-- EXEC sp_Calculate_Model5b_iBudget @BatchSize = 500;

BEGIN
    SET NOCOUNT ON;
    
    -- Create temporary table for results
    IF OBJECT_ID('tempdb..#Model5bResults') IS NOT NULL
        DROP TABLE #Model5bResults;
    
    CREATE TABLE #Model5bResults (
        CaseNo BIGINT,
        AssessID BIGINT,
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
        CalculationDate DATETIME DEFAULT GETDATE()
    );
    
    -- Model 5b Coefficients (from Table 4 in the document)
    DECLARE @Intercept DECIMAL(19, 4) = 27.5720;
    DECLARE @Live_ILSL DECIMAL(19, 4) = 35.8220;
    DECLARE @Live_RH1 DECIMAL(19, 4) = 90.6294;
    DECLARE @Live_RH2 DECIMAL(19, 4) = 131.7576;
    DECLARE @Live_RH3 DECIMAL(19, 4) = 209.4558;
    DECLARE @Live_RH4 DECIMAL(19, 4) = 267.0995;
    DECLARE @Age21_30 DECIMAL(19, 4) = 47.8473;
    DECLARE @Age31Plus DECIMAL(19, 4) = 48.9634;
    DECLARE @BSumCoeff DECIMAL(19, 4) = 0.4954;
    DECLARE @FHFSumCoeff DECIMAL(19, 4) = 0.6349;
    DECLARE @SLFSumCoeff DECIMAL(19, 4) = 2.0529;
    DECLARE @SLBSumCoeff DECIMAL(19, 4) = 1.4501;
    DECLARE @Q16Coeff DECIMAL(19, 4) = 2.4984;
    DECLARE @Q18Coeff DECIMAL(19, 4) = 5.8537;
    DECLARE @Q20Coeff DECIMAL(19, 4) = 2.6772;
    DECLARE @Q21Coeff DECIMAL(19, 4) = 2.7878;
    DECLARE @Q23Coeff DECIMAL(19, 4) = 6.3555;
    DECLARE @Q28Coeff DECIMAL(19, 4) = 2.2803;
    DECLARE @Q33Coeff DECIMAL(19, 4) = 1.2233;
    DECLARE @Q34Coeff DECIMAL(19, 4) = 2.1764;
    DECLARE @Q36Coeff DECIMAL(19, 4) = 2.6734;
    DECLARE @Q43Coeff DECIMAL(19, 4) = 1.9304;
    
    -- Process QSI assessments
    ;WITH QSIData AS (
        SELECT 
            q.CASENO,
            q.AssessID,
            q.REVIEWDATE,
            c.DOB,
            -- Calculate age
            DATEDIFF(YEAR, c.DOB, COALESCE(q.REVIEWDATE, GETDATE())) - 
                CASE 
                    WHEN DATEADD(YEAR, DATEDIFF(YEAR, c.DOB, COALESCE(q.REVIEWDATE, GETDATE())), c.DOB) > COALESCE(q.REVIEWDATE, GETDATE())
                    THEN 1 
                    ELSE 0 
                END AS Age,
            
            -- Determine Living Setting based on actual service codes in your database
            CASE 
                -- RH4: Special Medical Home Care (CTEP and Special Medical)
                WHEN EXISTS (
                    SELECT 1 FROM dbo.tbl_PlannedServices ps 
                    WHERE ps.CaseNo = q.CASENO 
                    AND ps.ServiceCode IN (
                        'S9122UC',     -- (4240) Special Medical Home Care
                        'T2025UC',     -- (4600) Enhanced Intensive Behavioral Residential Habilitation, Day
                        'T2025UCSE',   -- (4602) Enhanced Intensive Behavioral Residential Habilitation Medical, Day
                        'T2023UCSE',   -- (4603) Enhanced Intensive Behavioral Residential Habilitation Medical, Month
                        'T2023UCTG'    -- (4601) Enhanced Intensive Behavioral Residential Habilitation, Month
                    )
                    AND ps.PlannedServiceStatus IN ('Approved', 'Region Review Approved', 'State Review Approved')
                ) THEN 'RH4'
                
                -- RH3: Intensive Behavioral (Day Level 1-6)
                WHEN EXISTS (
                    SELECT 1 FROM dbo.tbl_PlannedServices ps 
                    WHERE ps.CaseNo = q.CASENO 
                    AND ps.ServiceCode IN (
                        'T2016UC',     -- (4500) Residential Habilitation - Intensive Behavioral - Day Level 1
                        'T2016UCHM',   -- (4501) Residential Habilitation - Intensive Behavioral - Day Level 2
                        'T2016UCHN',   -- (4502) Residential Habilitation - Intensive Behavioral - Day Level 3
                        'T2016UCHO',   -- (4503) Residential Habilitation - Intensive Behavioral - Day Level 4
                        'T2016UCHP',   -- (4504) Residential Habilitation - Intensive Behavioral - Day Level 5
                        'T2016UCSC'    -- (4505) Residential Habilitation - Intensive Behavioral - Day Level 6
                    )
                    AND ps.PlannedServiceStatus = 'Active'
                ) THEN 'RH3'
                
                -- RH2: Behavioral Focus
                WHEN EXISTS (
                    SELECT 1 FROM dbo.tbl_PlannedServices ps 
                    WHERE ps.CaseNo = q.CASENO 
                    AND ps.ServiceCode IN (
                        'T2020UC',     -- (4172) Residential Habilitation - Behavioral Focus - Minimal (day)
                        'T2020UCHI',   -- (4173) Residential Habilitation - Behavioral Focus - Moderate (day)
                        'T2020UCHM',   -- (4170) Residential Habilitation - Behavioral Focus - Extensive 1 (day)
                        'T2020UCHN',   -- (4171) Residential Habilitation - Behavioral Focus - Extensive 2 (day)
                        'T2023UCHM',   -- (4183) Residential Habilitation - Behavioral Focus - Minimal (month)
                        'T2023UCHN',   -- (4184) Residential Habilitation - Behavioral Focus - Moderate (month)
                        'T2023UCHO',   -- (4181) Residential Habilitation - Behavioral Focus - Extensive 1 (month)
                        'T2023UCHP'    -- (4182) Residential Habilitation - Behavioral Focus - Extensive 2 (month)
                    )
                    AND ps.PlannedServiceStatus = 'Active'
                ) THEN 'RH2'
                
                -- RH1: Standard Residential Habilitation (including Live In)
                WHEN EXISTS (
                    SELECT 1 FROM dbo.tbl_PlannedServices ps 
                    WHERE ps.CaseNo = q.CASENO 
                    AND ps.ServiceCode IN (
                        'H0043UC',     -- (4176) Residential Habilitation - Basic (day)
                        'H0043UCHI',   -- (4177) Residential Habilitation - Minimal (day)
                        'H0043UCHM',   -- (4178) Residential Habilitation - Moderate (day)
                        'H0043UCHN',   -- (4179) Residential Habilitation - Extensive 1 (day)
                        'H0043UCHO',   -- (4180) Residential Habilitation - Extensive 2 (day)
                        'H0043UCSC',   -- (4175) Res.Hab. - Live In
                        'T2023UC',     -- (4186) Residential Habilitation - Basic (month)
                        'T2023UCSC',   -- (4187) Residential Habilitation - Minimal (month)
                        'T2023UCU4',   -- (4188) Residential Habilitation - Moderate (month)
                        'T2023UCU6',   -- (4189) Residential Habilitation - Extensive 1 (month)
                        'T2023UCU9'    -- (4190) Residential Habilitation - Extensive 2 (month)
                    )
                    AND ps.PlannedServiceStatus = 'Active'
                ) THEN 'RH1'
                
                -- ILSL: Supported Living Coaching (97535UC is the main code)
                WHEN EXISTS (
                    SELECT 1 FROM dbo.tbl_PlannedServices ps 
                    WHERE ps.CaseNo = q.CASENO 
                    AND ps.ServiceCode = '97535UC' -- (4290) Supported Living Coaching
                    AND ps.PlannedServiceStatus = 'Active'
                ) THEN 'ILSL'
                
                -- Check RESIDENCETYPE as fallback
                WHEN c.RESIDENCETYPE LIKE '%Residential%' THEN 'RH1'
                WHEN c.RESIDENCETYPE LIKE '%Behavior%' THEN 'RH2'
                WHEN c.RESIDENCETYPE LIKE '%Intensive%' THEN 'RH3'
                WHEN c.RESIDENCETYPE LIKE '%Special Medical%' THEN 'RH4'
                WHEN c.RESIDENCETYPE LIKE '%Support%' OR c.RESIDENCETYPE LIKE '%Independent%' THEN 'ILSL'
                
                -- Default to Family Home if no residential services
                ELSE 'FH'
            END AS LivingSetting,
            
            -- Convert QSI scores to integers (they're stored as varchar)
            CAST(ISNULL(q.Q14, '0') AS INT) AS Q14,
            CAST(ISNULL(q.Q15, '0') AS INT) AS Q15,
            CAST(ISNULL(q.Q16, '0') AS INT) AS Q16,
            CAST(ISNULL(q.Q17, '0') AS INT) AS Q17,
            CAST(ISNULL(q.Q18, '0') AS INT) AS Q18,
            CAST(ISNULL(q.Q19, '0') AS INT) AS Q19,
            CAST(ISNULL(q.Q20, '0') AS INT) AS Q20,
            CAST(ISNULL(q.Q21, '0') AS INT) AS Q21,
            CAST(ISNULL(q.Q22, '0') AS INT) AS Q22,
            CAST(ISNULL(q.Q23, '0') AS INT) AS Q23,
            CAST(ISNULL(q.Q24, '0') AS INT) AS Q24,
            CAST(ISNULL(q.Q25, '0') AS INT) AS Q25,
            CAST(ISNULL(q.Q26, '0') AS INT) AS Q26,
            CAST(ISNULL(q.Q27, '0') AS INT) AS Q27,
            CAST(ISNULL(q.Q28, '0') AS INT) AS Q28,
            CAST(ISNULL(q.Q29, '0') AS INT) AS Q29,
            CAST(ISNULL(q.Q30, '0') AS INT) AS Q30,
            CAST(ISNULL(q.Q32, '0') AS INT) AS Q32,
            CAST(ISNULL(q.Q33, '0') AS INT) AS Q33,
            CAST(ISNULL(q.Q34, '0') AS INT) AS Q34,
            CAST(ISNULL(q.Q35, '0') AS INT) AS Q35,
            CAST(ISNULL(q.Q36, '0') AS INT) AS Q36,
            CAST(ISNULL(q.Q37, '0') AS INT) AS Q37,
            CAST(ISNULL(q.Q38, '0') AS INT) AS Q38,
            CAST(ISNULL(q.Q39, '0') AS INT) AS Q39,
            CAST(ISNULL(q.Q40, '0') AS INT) AS Q40,
            CAST(ISNULL(q.Q41, '0') AS INT) AS Q41,
            CAST(ISNULL(q.Q42, '0') AS INT) AS Q42,
            CAST(ISNULL(q.Q43, '0') AS INT) AS Q43,
            CAST(ISNULL(q.Q44, '0') AS INT) AS Q44,
            CAST(ISNULL(q.Q45, '0') AS INT) AS Q45,
            CAST(ISNULL(q.Q46, '0') AS INT) AS Q46,
            CAST(ISNULL(q.Q47, '0') AS INT) AS Q47,
            CAST(ISNULL(q.Q48, '0') AS INT) AS Q48,
            CAST(ISNULL(q.Q49, '0') AS INT) AS Q49,
            CAST(ISNULL(q.Q50, '0') AS INT) AS Q50
        FROM dbo.tbl_QSIAssessments q
        INNER JOIN dbo.tbl_Consumers c ON q.CASENO = c.CASENO
        WHERE (@CaseNo IS NULL OR q.CASENO = @CaseNo)
            -- Remove strict STATUS filter - check for any non-null status
            AND q.STATUS IS NOT NULL
            -- Get most recent assessment per consumer
            AND q.AssessID = (
                SELECT TOP 1 q2.AssessID 
                FROM dbo.tbl_QSIAssessments q2 
                WHERE q2.CASENO = q.CASENO 
                    AND q2.STATUS IS NOT NULL
                ORDER BY q2.REVIEWDATE DESC, q2.AssessID DESC
            )
    ),
    CalculatedSums AS (
        SELECT 
            *,
            -- Calculate FSum (Functional Sum: Q14-Q24)
            Q14 + Q15 + Q16 + Q17 + Q18 + Q19 + Q20 + Q21 + Q22 + Q23 + Q24 AS FSum,
            -- Calculate BSum (Behavioral Sum: Q25-Q30)
            Q25 + Q26 + Q27 + Q28 + Q29 + Q30 AS BSum,
            -- Calculate PSum (Physical Sum: Q32-Q50)
            Q32 + Q33 + Q34 + Q35 + Q36 + Q37 + Q38 + Q39 + Q40 + 
            Q41 + Q42 + Q43 + Q44 + Q45 + Q46 + Q47 + Q48 + Q49 + Q50 AS PSum,
            -- Determine age group
            CASE 
                WHEN Age < 21 THEN 'Under21'
                WHEN Age >= 21 AND Age <= 30 THEN 'Age21-30'
                ELSE 'Age31Plus'
            END AS AgeGroup
        FROM QSIData
    ),
    InteractionTerms AS (
        SELECT 
            *,
            -- Calculate interaction terms
            CASE WHEN LivingSetting = 'FH' THEN FSum ELSE 0 END AS FHFSum,
            CASE WHEN LivingSetting = 'ILSL' THEN FSum ELSE 0 END AS SLFSum,
            CASE WHEN LivingSetting = 'ILSL' THEN BSum ELSE 0 END AS SLBSum
        FROM CalculatedSums
    )
    INSERT INTO #Model5bResults (
        CaseNo, AssessID, ReviewDate, LivingSetting, Age, AgeGroup, 
        BSum, FSum, PSum, FHFSum, SLFSum, SLBSum, 
        SqrtPrediction, PredictedBudget
    )
    SELECT 
        CASENO,
        AssessID,
        REVIEWDATE,
        LivingSetting,
        Age,
        AgeGroup,
        BSum,
        FSum,
        PSum,
        FHFSum,
        SLFSum,
        SLBSum,
        -- Calculate square-root scale prediction
        @Intercept +
        -- Living setting effects (FH is reference level with coefficient 0)
        CASE LivingSetting
            WHEN 'ILSL' THEN @Live_ILSL
            WHEN 'RH1' THEN @Live_RH1
            WHEN 'RH2' THEN @Live_RH2
            WHEN 'RH3' THEN @Live_RH3
            WHEN 'RH4' THEN @Live_RH4
            ELSE 0 -- FH (reference)
        END +
        -- Age effects (Under 21 is reference level with coefficient 0)
        CASE AgeGroup
            WHEN 'Age21-30' THEN @Age21_30
            WHEN 'Age31Plus' THEN @Age31Plus
            ELSE 0 -- Under 21 (reference)
        END +
        -- Sum scores and interaction terms
        (@BSumCoeff * BSum) +
        (@FHFSumCoeff * FHFSum) +
        (@SLFSumCoeff * SLFSum) +
        (@SLBSumCoeff * SLBSum) +
        -- QSI question effects
        (@Q16Coeff * Q16) +
        (@Q18Coeff * Q18) +
        (@Q20Coeff * Q20) +
        (@Q21Coeff * Q21) +
        (@Q23Coeff * Q23) +
        (@Q28Coeff * Q28) +
        (@Q33Coeff * Q33) +
        (@Q34Coeff * Q34) +
        (@Q36Coeff * Q36) +
        (@Q43Coeff * Q43) AS SqrtPrediction,
        -- Square the prediction to get budget amount
        POWER(
            @Intercept +
            CASE LivingSetting
                WHEN 'ILSL' THEN @Live_ILSL
                WHEN 'RH1' THEN @Live_RH1
                WHEN 'RH2' THEN @Live_RH2
                WHEN 'RH3' THEN @Live_RH3
                WHEN 'RH4' THEN @Live_RH4
                ELSE 0
            END +
            CASE AgeGroup
                WHEN 'Age21-30' THEN @Age21_30
                WHEN 'Age31Plus' THEN @Age31Plus
                ELSE 0
            END +
            (@BSumCoeff * BSum) +
            (@FHFSumCoeff * FHFSum) +
            (@SLFSumCoeff * SLFSum) +
            (@SLBSumCoeff * SLBSum) +
            (@Q16Coeff * Q16) +
            (@Q18Coeff * Q18) +
            (@Q20Coeff * Q20) +
            (@Q21Coeff * Q21) +
            (@Q23Coeff * Q23) +
            (@Q28Coeff * Q28) +
            (@Q33Coeff * Q33) +
            (@Q34Coeff * Q34) +
            (@Q36Coeff * Q36) +
            (@Q43Coeff * Q43), 
            2
        ) AS PredictedBudget
    FROM InteractionTerms;
    
    -- Return results
    -- Join with consumer contacts if names are needed
    SELECT 
        r.CaseNo,
        -- Get primary contact names if available
        cc.FIRSTNAME AS ContactFirstName,
        cc.LASTNAME AS ContactLastName,
        c.PRIMARYWORKER AS PrimaryWorker,
        c.County,
        c.Region,
        c.RESIDENCETYPE,
        r.AssessID,
        r.ReviewDate,
        r.LivingSetting AS CalculatedLivingSetting,
        r.Age,
        r.AgeGroup,
        r.BSum AS BehavioralSum,
        r.FSum AS FunctionalSum,
        r.PSum AS PhysicalSum,
        r.FHFSum AS FamilyHomeFunctionalInteraction,
        r.SLFSum AS ILSLFunctionalInteraction,
        r.SLBSum AS ILSLBehavioralInteraction,
        r.SqrtPrediction AS SquareRootPrediction,
        r.PredictedBudget AS PredictedBudgetAmount,
        r.CalculationDate
    FROM #Model5bResults r
    INNER JOIN dbo.tbl_Consumers c ON r.CaseNo = c.CASENO
    LEFT JOIN dbo.tbl_ConsumerContacts cc ON c.CASENO = cc.CASENO 
        AND cc.Active = 1
        AND cc.RELATIONSHIP IN ('Self', 'Legal Guardian', 'Parent')
    ORDER BY r.CaseNo;
    
    -- Store results in permanent table for tracking with RunID
    IF OBJECT_ID('dbo.tbl_Model5b_Predictions', 'U') IS NOT NULL
    BEGIN
        DECLARE @RunID bigint
        SELECT @RunID = MAX(RunID) FROM tbl_Model5b_Predictions
        SET @RunID = ISNULL(@RunID, 0) + 1
        
        INSERT INTO dbo.tbl_Model5b_Predictions
        SELECT *, GETDATE() AS CreatedDate, SYSTEM_USER AS CreatedBy, @RunID AS RunID
        FROM #Model5bResults;
    END
    
    -- Clean up
    DROP TABLE #Model5bResults;
END
GO

-- Grant execute permissions
--GRANT EXECUTE ON [dbo].[sp_Calculate_Model5b_iBudget] TO [APD_Users];
--GO

-- =============================================
-- SUPPORTING TABLE FOR STORING PREDICTIONS:
-- =============================================
/*
CREATE TABLE dbo.tbl_Model5b_Predictions (
    PredictionID BIGINT IDENTITY(1,1) PRIMARY KEY,
    RunID BIGINT NOT NULL,
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
    INDEX IX_RunID (RunID),
    INDEX IX_CaseNo (CaseNo),
    INDEX IX_AssessID (AssessID),
    INDEX IX_CalculationDate (CalculationDate)
);
*/

-- =============================================
-- HELPER FUNCTION: Get Living Setting from Services
-- This would need to be customized based on actual business rules
-- =============================================
/*
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
*/