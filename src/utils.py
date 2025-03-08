# utils.py
import logging
logger = logging.getLogger(__name__)

def ensure_mitsuba_variant(variant='cuda_ad_rgb'):
    """
    Ensure the correct Mitsuba variant is set.
    
    Args:
        variant (str): The Mitsuba variant to set (default: 'cuda_ad_rgb')
        
    Returns:
        str: The current Mitsuba variant after setting
        
    Raises:
        RuntimeError: If the variant cannot be set correctly
        ImportError: If Mitsuba is not installed or importable
    """
    try:
        import mitsuba
        current = mitsuba.variant()
        logger.debug(f"Current Mitsuba variant before check: {current}")
        
        if current != variant:
            logger.info(f"Setting Mitsuba variant from {current} to {variant}")
            mitsuba.set_variant(variant)
            current = mitsuba.variant()
            
            # Verify the variant was set correctly
            if current != variant:
                raise RuntimeError(f"Failed to set Mitsuba variant to {variant}. Current: {current}")
        
        logger.debug(f"Mitsuba variant verified: {current}")
        return current
    
    except ImportError:
        logger.error("Mitsuba not installed or not importable")
        raise
    except Exception as e:
        logger.error(f"Error ensuring Mitsuba variant: {str(e)}")
        raise RuntimeError(f"Failed to set Mitsuba variant: {str(e)}")