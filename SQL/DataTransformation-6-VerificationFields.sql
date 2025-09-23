USE [APD]
GO

-- Script to add ISFCal column to tbl_EZBudget and tbl_QSIAssessments
-- This column will store the Individual Support Fund calculation based on Model 5b, ISFCal  :)

-- Add ISFCal column to tbl_EZBudget
IF NOT EXISTS (
    SELECT * 
    FROM sys.columns 
    WHERE object_id = OBJECT_ID(N'[dbo].[tbl_EZBudget]') 
    AND name = 'ISFCal'
)
BEGIN
    ALTER TABLE [dbo].[tbl_EZBudget]
    ADD ISFCal float NULL;
    
    PRINT 'ISFCal column added to tbl_EZBudget';
END
ELSE
BEGIN
    PRINT 'ISFCal column already exists in tbl_EZBudget';
END
GO

-- Add ISFCal column to tbl_QSIAssessments
IF NOT EXISTS (
    SELECT * 
    FROM sys.columns 
    WHERE object_id = OBJECT_ID(N'[dbo].[tbl_QSIAssessments]') 
    AND name = 'ISFCal'
)
BEGIN
    ALTER TABLE [dbo].[tbl_QSIAssessments]
    ADD ISFCal float NULL;
    
    PRINT 'ISFCal column added to tbl_QSIAssessments';
END
ELSE
BEGIN
    PRINT 'ISFCal column already exists in tbl_QSIAssessments';
END
GO

-- Add index on ISFCal columns for better query performance (optional)
IF NOT EXISTS (
    SELECT * 
    FROM sys.indexes 
    WHERE object_id = OBJECT_ID(N'[dbo].[tbl_EZBudget]') 
    AND name = N'IX_tbl_EZBudget_ISFCal'
)
BEGIN
    CREATE NONCLUSTERED INDEX IX_tbl_EZBudget_ISFCal 
    ON [dbo].[tbl_EZBudget] (ISFCal);
    
    PRINT 'Index IX_tbl_EZBudget_ISFCal created';
END
GO

IF NOT EXISTS (
    SELECT * 
    FROM sys.indexes 
    WHERE object_id = OBJECT_ID(N'[dbo].[tbl_QSIAssessments]') 
    AND name = N'IX_tbl_QSIAssessments_ISFCal'
)
BEGIN
    CREATE NONCLUSTERED INDEX IX_tbl_QSIAssessments_ISFCal 
    ON [dbo].[tbl_QSIAssessments] (ISFCal);
    
    PRINT 'Index IX_tbl_QSIAssessments_ISFCal created';
END
GO

-- Verify the columns were added successfully
SELECT 
    t.name AS TableName,
    c.name AS ColumnName,
    ty.name AS DataType,
    c.is_nullable
FROM sys.columns c
INNER JOIN sys.tables t ON c.object_id = t.object_id
INNER JOIN sys.types ty ON c.user_type_id = ty.user_type_id
WHERE t.name IN ('tbl_EZBudget', 'tbl_QSIAssessments')
    AND c.name = 'ISFCal'
ORDER BY t.name;