USE [APD]
GO

-- Check log space usage
DBCC SQLPERF(LOGSPACE);

-- If in SIMPLE recovery mode, you can shrink the log
-- First backup the database!
ALTER DATABASE [APD] SET RECOVERY SIMPLE;
GO
CHECKPOINT;
GO
DBCC SHRINKFILE (APD_Data_ISF_log, 1);  -- Replace 'APD_log' with your actual log file name
GO


-- Step 1: Add the Age column to tbl_Claims_MMIS
IF NOT EXISTS (
    SELECT * 
    FROM sys.columns 
    WHERE object_id = OBJECT_ID(N'[dbo].[tbl_Claims_MMIS]') 
    AND name = 'Age'
)
BEGIN
    ALTER TABLE [dbo].[tbl_Claims_MMIS]
    ADD Age int NULL;
    
    PRINT 'Age column added to tbl_Claims_MMIS';
END
ELSE
BEGIN
    PRINT 'Age column already exists in tbl_Claims_MMIS';
END
GO

USE [APD]
GO

-- Step 2: Populate the Age field in batches
DECLARE @BatchSize INT = 100000;  -- Adjust batch size as needed
DECLARE @RowsUpdated INT = 0;
DECLARE @TotalRows INT = 0;
DECLARE @ProgressCount INT = 0;

-- Get total count of records to update
SELECT @TotalRows = COUNT(*)
FROM [dbo].[tbl_Claims_MMIS] cm
INNER JOIN [dbo].[tbl_Consumers] c ON cm.CaseNo = c.CASENO
WHERE c.DOB IS NOT NULL 
  AND cm.ServiceDate IS NOT NULL
  AND cm.Age IS NULL;  -- Only update records that haven't been updated yet

PRINT 'Total records to update: ' + CAST(@TotalRows AS VARCHAR(10));

-- Update in batches
WHILE 1 = 1
BEGIN
    -- Update a batch of records
    UPDATE TOP (@BatchSize) cm
    SET cm.Age = DATEDIFF(YEAR, c.DOB, cm.ServiceDate) - 
                  CASE 
                      WHEN MONTH(cm.ServiceDate) < MONTH(c.DOB) OR 
                           (MONTH(cm.ServiceDate) = MONTH(c.DOB) AND DAY(cm.ServiceDate) < DAY(c.DOB))
                      THEN 1 
                      ELSE 0 
                  END
    FROM [dbo].[tbl_Claims_MMIS] cm
    INNER JOIN [dbo].[tbl_Consumers] c ON cm.CaseNo = c.CASENO
    WHERE c.DOB IS NOT NULL 
      AND cm.ServiceDate IS NOT NULL
      AND cm.Age IS NULL;  -- Only update records that haven't been updated yet
    
    -- Check how many rows were updated
    SET @RowsUpdated = @@ROWCOUNT;
    
    -- Exit loop if no more rows to update
    IF @RowsUpdated = 0
        BREAK;
    
    -- Get progress count (store in variable first)
    SELECT @ProgressCount = COUNT(*) FROM [dbo].[tbl_Claims_MMIS] WHERE Age IS NOT NULL;
    
    -- Print progress
    PRINT 'Updated ' + CAST(@RowsUpdated AS VARCHAR(10)) + ' records. Progress: ' + 
          CAST(@ProgressCount AS VARCHAR(10)) + ' / ' + CAST(@TotalRows AS VARCHAR(10));
    
    -- Optional: Add a small delay to reduce server load
    WAITFOR DELAY '00:00:01';  -- 1 second delay
    
    -- Optional: Checkpoint to clear the log (if database is in SIMPLE recovery mode)
    CHECKPOINT;  -- Uncommented this since you're now in SIMPLE recovery mode
END

-- Get final count
DECLARE @FinalCount INT = 0;

SELECT @FinalCount = COUNT(*) FROM [dbo].[tbl_Claims_MMIS] WHERE Age IS NOT NULL;

PRINT 'Age field population completed. Total rows updated: ' + CAST(@FinalCount AS VARCHAR(10));
GO


-- Step 3: Create composite index on CaseNo and Age
IF NOT EXISTS (
    SELECT * 
    FROM sys.indexes 
    WHERE object_id = OBJECT_ID(N'[dbo].[tbl_Claims_MMIS]') 
    AND name = N'IX_tbl_Claims_MMIS_CaseNo_Age'
)
BEGIN
    CREATE NONCLUSTERED INDEX IX_tbl_Claims_MMIS_CaseNo_Age 
    ON [dbo].[tbl_Claims_MMIS] (CaseNo, Age)
    INCLUDE (ServiceDate);  -- Optional: include ServiceDate for covering index
    
    PRINT 'Composite index IX_tbl_Claims_MMIS_CaseNo_Age created';
END
ELSE
BEGIN
    PRINT 'Index IX_tbl_Claims_MMIS_CaseNo_Age already exists';
END
GO

-- Step 4: Verify the results
-- Check some sample data
SELECT TOP 100
    cm.CaseNo,
    c.DOB,
    cm.ServiceDate,
    cm.Age,
    DATEDIFF(YEAR, c.DOB, cm.ServiceDate) AS SimpleDateDiff
FROM [dbo].[tbl_Claims_MMIS] cm
INNER JOIN [dbo].[tbl_Consumers] c ON cm.CaseNo = c.CASENO
WHERE cm.Age IS NOT NULL
ORDER BY cm.CaseNo;

-- Check statistics
SELECT 
    COUNT(*) AS TotalRecords,
    COUNT(Age) AS RecordsWithAge,
    COUNT(*) - COUNT(Age) AS RecordsWithoutAge,
    MIN(Age) AS MinAge,
    MAX(Age) AS MaxAge,
    AVG(Age) AS AvgAge
FROM [dbo].[tbl_Claims_MMIS];



