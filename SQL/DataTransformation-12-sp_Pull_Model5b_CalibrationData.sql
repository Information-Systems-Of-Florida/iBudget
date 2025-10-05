USE [APD]
GO

-- Drop procedure if it exists
IF EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[sp_Pull_Model5b_CalibrationData]') AND type in (N'P', N'PC'))
DROP PROCEDURE [dbo].[sp_Pull_Model5b_CalibrationData]
GO

CREATE PROCEDURE [dbo].[sp_Pull_Model5b_CalibrationData]
    @StartDate datetime = '2024-07-01',
    @EndDate datetime = '2025-06-30',
    @Debug bit = 0
AS
/*

Update bussines rule: 

[ServiceDate] between September 1, 2023 and August 31, 2024.
[PaidAmt]: Aggregate in the period of [ServiceDate] +  one year for services between September 1, 2023 and August 31, 2024

*/



BEGIN
    SET NOCOUNT ON;
    
    -- Create temp table for latest QSI assessments
    CREATE TABLE #LatestQSI (
        CaseNo bigint,
        AssessID bigint,
        ReviewDate datetime,
        RowNum int
    );
    
    -- Get latest QSI assessment for each consumer who has claims in the period
    WITH ClaimConsumers AS (
        SELECT DISTINCT CaseNo
        FROM [dbo].[tbl_Claims_MMIS]
        WHERE ServiceDate >= @StartDate 
        AND ServiceDate < @EndDate
    ),
    RankedQSI AS (
        SELECT 
            q.CaseNo,
            q.AssessID,
            q.ReviewDate,
            ROW_NUMBER() OVER (PARTITION BY q.CaseNo ORDER BY q.ReviewDate DESC, q.AssessID DESC) as RowNum
        FROM [dbo].[tbl_QSIAssessments] q
        INNER JOIN ClaimConsumers c ON q.CaseNo = c.CaseNo
    )
    INSERT INTO #LatestQSI
    SELECT CaseNo, AssessID, ReviewDate, RowNum
    FROM RankedQSI
    WHERE RowNum = 1;
    
    -- Main query to pull all data needed for calibration
    SELECT 
        -- Consumer identifiers
        q.CaseNo,
        q.AssessID,
        q.ReviewDate,
        
        -- Demographics from tbl_Consumers
        c.DOB,
        DATEDIFF(YEAR, c.DOB, GETDATE()) AS CurrentAge,
        CASE 
            WHEN DATEDIFF(YEAR, c.DOB, GETDATE()) BETWEEN 21 AND 30 THEN 'Age21-30'
            WHEN DATEDIFF(YEAR, c.DOB, GETDATE()) >= 31 THEN 'Age31+'
            ELSE 'Under21'
        END AS AgeGroup,
        
        -- Living Setting with standardized mapping
        c.RESIDENCETYPE,
        CASE 
            -- Family Home (reference category)
            WHEN c.RESIDENCETYPE LIKE '%Family Home%' THEN 'FH'
            
            -- Independent Living/Supported Living
            WHEN c.RESIDENCETYPE LIKE '%Independent Living%' 
                OR c.RESIDENCETYPE LIKE '%Supported Living%' THEN 'ILSL'
            
            -- Standard Residential (RH1) - Small Group Homes and Foster Homes
            WHEN c.RESIDENCETYPE LIKE '%Small Group Home%' 
                OR c.RESIDENCETYPE LIKE '%Foster Home%'
                OR c.RESIDENCETYPE LIKE '%Adult Family Care Home%' THEN 'RH1'
            
            -- Behavior Focus (RH2) - Large Group Homes
            WHEN c.RESIDENCETYPE LIKE '%Large Group Home%' THEN 'RH2'
            
            -- Intensive Behavior (RH3) - ICF/DD, Assisted Living, Nursing Home
            WHEN c.RESIDENCETYPE LIKE '%ICF/DD%'
                OR c.RESIDENCETYPE LIKE '%Assisted Living%'
                OR c.RESIDENCETYPE LIKE '%Nursing Home%' THEN 'RH3'
            
            -- CTEP/Special Medical (RH4) - Centers, Hospitals, Special Facilities
            WHEN c.RESIDENCETYPE LIKE '%Developmental Disabilities Center%'
                OR c.RESIDENCETYPE LIKE '%Residential Habilitation Center%'
                OR c.RESIDENCETYPE LIKE '%Hospital%'
                OR c.RESIDENCETYPE LIKE '%Defendant Program%'
                OR c.RESIDENCETYPE LIKE '%Commitment Facility%'
                OR c.RESIDENCETYPE LIKE '%Mental Health%' THEN 'RH4'
            
            -- Other/Unknown
            ELSE 'OTHER'
        END AS LivingSettingCategory,
        
        -- QSI Questions needed for Model 5b
        -- Note: Only including questions that exist and are needed for the model
        q.Q14, q.Q15, q.Q16, q.Q17, q.Q18, q.Q19, q.Q20,
        q.Q21, q.Q22, q.Q23, q.Q24, q.Q25, q.Q26, q.Q27,
        q.Q28, q.Q29, q.Q30, 
        q.Q33, q.Q34, q.Q36, q.Q43,
        
        -- Calculated sums for Model5b
        -- Behavioral Sum (Q25-Q30)
        (ISNULL(TRY_CAST(q.Q25 AS float), 0) + 
         ISNULL(TRY_CAST(q.Q26 AS float), 0) + 
         ISNULL(TRY_CAST(q.Q27 AS float), 0) + 
         ISNULL(TRY_CAST(q.Q28 AS float), 0) + 
         ISNULL(TRY_CAST(q.Q29 AS float), 0) + 
         ISNULL(TRY_CAST(q.Q30 AS float), 0)) AS BSum,
        
        -- Functional Sum (Q14-Q24)
        (ISNULL(TRY_CAST(q.Q14 AS float), 0) + 
         ISNULL(TRY_CAST(q.Q15 AS float), 0) + 
         ISNULL(TRY_CAST(q.Q16 AS float), 0) + 
         ISNULL(TRY_CAST(q.Q17 AS float), 0) + 
         ISNULL(TRY_CAST(q.Q18 AS float), 0) + 
         ISNULL(TRY_CAST(q.Q19 AS float), 0) + 
         ISNULL(TRY_CAST(q.Q20 AS float), 0) + 
         ISNULL(TRY_CAST(q.Q21 AS float), 0) + 
         ISNULL(TRY_CAST(q.Q22 AS float), 0) + 
         ISNULL(TRY_CAST(q.Q23 AS float), 0) + 
         ISNULL(TRY_CAST(q.Q24 AS float), 0)) AS FSum,
        
        -- Total Claims Amount
        claims.TotalClaims,
        claims.ClaimCount,
        
        -- Data Quality Flag
        CASE 
            WHEN q.Q14 IS NULL OR q.Q15 IS NULL OR q.Q16 IS NULL OR q.Q17 IS NULL OR
                 q.Q18 IS NULL OR q.Q19 IS NULL OR q.Q20 IS NULL OR q.Q21 IS NULL OR
                 q.Q22 IS NULL OR q.Q23 IS NULL OR q.Q24 IS NULL OR q.Q25 IS NULL OR
                 q.Q26 IS NULL OR q.Q27 IS NULL OR q.Q28 IS NULL OR q.Q29 IS NULL OR
                 q.Q30 IS NULL OR q.Q33 IS NULL OR q.Q34 IS NULL OR q.Q36 IS NULL OR
                 q.Q43 IS NULL THEN 1
            ELSE 0
        END AS HasMissingQSI,
        
        CASE 
            WHEN c.DOB IS NULL OR c.RESIDENCETYPE IS NULL THEN 1
            ELSE 0
        END AS HasMissingDemographics
        
    FROM #LatestQSI lq
    INNER JOIN [dbo].[tbl_QSIAssessments] q ON lq.CaseNo = q.CaseNo AND lq.AssessID = q.AssessID
    INNER JOIN [dbo].[tbl_Consumers] c ON q.CaseNo = c.CASENO
    INNER JOIN (
        -- Aggregate claims for the period
        SELECT 
            CaseNo,
            SUM(PaidAmt) AS TotalClaims,
            COUNT(*) AS ClaimCount
        FROM [dbo].[tbl_Claims_MMIS]
        WHERE ServiceDate >= @StartDate 
        AND ServiceDate < @EndDate
        GROUP BY CaseNo
    ) claims ON q.CaseNo = claims.CaseNo
    ORDER BY q.CaseNo;
    
    -- Debug information
    IF @Debug = 1
    BEGIN
        SELECT 
            'Total Consumers with Claims' AS Metric,
            COUNT(DISTINCT CaseNo) AS Count
        FROM [dbo].[tbl_Claims_MMIS]
        WHERE ServiceDate >= @StartDate 
        AND ServiceDate < @EndDate;
        
        SELECT 
            'Consumers with QSI Assessments' AS Metric,
            COUNT(*) AS Count
        FROM #LatestQSI;
        
        SELECT 
            'Living Setting Distribution' AS Metric,
            LivingSettingCategory,
            COUNT(*) AS Count
        FROM (
            SELECT 
                CASE 
                    WHEN c.RESIDENCETYPE LIKE '%Family Home%' THEN 'FH'
                    WHEN c.RESIDENCETYPE LIKE '%Independent Living%' 
                        OR c.RESIDENCETYPE LIKE '%Supported Living%' THEN 'ILSL'
                    WHEN c.RESIDENCETYPE LIKE '%Small Group Home%' 
                        OR c.RESIDENCETYPE LIKE '%Foster Home%'
                        OR c.RESIDENCETYPE LIKE '%Adult Family Care Home%' THEN 'RH1'
                    WHEN c.RESIDENCETYPE LIKE '%Large Group Home%' THEN 'RH2'
                    WHEN c.RESIDENCETYPE LIKE '%ICF/DD%'
                        OR c.RESIDENCETYPE LIKE '%Assisted Living%'
                        OR c.RESIDENCETYPE LIKE '%Nursing Home%' THEN 'RH3'
                    WHEN c.RESIDENCETYPE LIKE '%Developmental Disabilities Center%'
                        OR c.RESIDENCETYPE LIKE '%Residential Habilitation Center%'
                        OR c.RESIDENCETYPE LIKE '%Hospital%'
                        OR c.RESIDENCETYPE LIKE '%Defendant Program%'
                        OR c.RESIDENCETYPE LIKE '%Commitment Facility%'
                        OR c.RESIDENCETYPE LIKE '%Mental Health%' THEN 'RH4'
                    ELSE 'OTHER'
                END AS LivingSettingCategory
            FROM #LatestQSI lq
            INNER JOIN [dbo].[tbl_Consumers] c ON lq.CaseNo = c.CASENO
        ) AS LivingSettings
        GROUP BY LivingSettingCategory;
    END
    
    -- Clean up
    DROP TABLE #LatestQSI;
    
END
GO

-- Grant execute permissions
GRANT EXECUTE ON [dbo].[sp_Pull_Model5b_CalibrationData] TO [public]
GO