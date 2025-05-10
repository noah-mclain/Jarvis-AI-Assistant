# Jarvis AI Assistant - Codebase Cleanup Plan

## Files to Delete

The following files are redundant and can be safely removed:

### 1. Fixed/Compatibility Files

- `src/generative_ai_module/fixed_run_finetune.py` - Functionality consolidated into `deepseek_training.py`

## Files to Modify

### 1. Update `__init__.py`

- Ensure all imports are correctly pointing to consolidated files
- Remove any imports of files that no longer exist
- Add clear documentation about which files contain which functionality

### 2. Consolidate Environment Setup

- Move all environment setup scripts from `src/generative_ai_module/` to `setup/`
- Create a single entry point for environment setup

## Logical Checks

### 1. DeepSeek-related Code

- Ensure `deepseek_training.py` and `consolidated_generation_pipeline.py` don't have conflicting functionality
- Check that all necessary DeepSeek functionality is properly exposed in `__init__.py`

### 2. Import Paths

- Verify all import paths are correct and use relative imports where appropriate
- Ensure no circular imports exist

### 3. Function Calls

- Check that all function calls refer to functions that exist
- Update any calls to functions in removed files to point to their new locations

## Documentation Updates

- Update main README.md with the new file structure
- Create a clear guide to which files contain which functionality
- Document the consolidated structure

## Implementation Steps

1. Back up the current state (already done in the backup directories)
2. Delete redundant files
3. Update imports in affected files
4. Run tests to ensure everything still works
5. Update documentation
