"""
Vision System Configuration Management
Centralized configuration with validation and persistence.
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict, field
from enum import Enum

# Import our format enums
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from capture.simple_screen_capture import ImageFormat, CompressionLevel


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class CaptureConfig:
    """Screen capture configuration."""
    default_format: ImageFormat = ImageFormat.PNG
    default_quality: CompressionLevel = CompressionLevel.HIGH
    max_file_size_kb: int = 2048
    auto_optimize_format: bool = True
    capture_timeout_seconds: float = 5.0
    
    def __post_init__(self):
        # Convert string values back to enums if loaded from JSON
        if isinstance(self.default_format, str):
            self.default_format = ImageFormat(self.default_format)
        if isinstance(self.default_quality, (str, int)):
            if isinstance(self.default_quality, str):
                self.default_quality = CompressionLevel(self.default_quality)
            else:
                # Handle integer values
                quality_map = {30: CompressionLevel.LOW, 60: CompressionLevel.MEDIUM, 
                             85: CompressionLevel.HIGH, 100: CompressionLevel.LOSSLESS}
                self.default_quality = quality_map.get(self.default_quality, CompressionLevel.HIGH)


@dataclass
class ServiceConfig:
    """Service configuration."""
    stream_interval_seconds: float = 4.0
    max_archive_files: int = 500
    max_archive_age_days: int = 7
    service_check_interval_seconds: float = 0.5
    auto_start_streaming: bool = True
    log_level: LogLevel = LogLevel.INFO
    
    def __post_init__(self):
        if isinstance(self.log_level, str):
            self.log_level = LogLevel(self.log_level)


@dataclass
class SecurityConfig:
    """Security configuration."""
    use_secure_storage: bool = True
    restrict_file_permissions: bool = True
    enable_acl_security: bool = True  # Windows ACL security
    max_session_age_hours: int = 24
    clean_temp_files: bool = True
    
    
@dataclass
class StorageConfig:
    """Storage configuration."""
    base_directory: Optional[str] = None  # None = auto-select secure location
    session_subdirectory: str = "claude_session"
    archive_subdirectory: str = "archive"
    service_subdirectory: str = "service"
    use_compression_for_archives: bool = True


@dataclass
class VisionSystemConfig:
    """Complete vision system configuration."""
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    service: ServiceConfig = field(default_factory=ServiceConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    
    # Metadata
    config_version: str = "1.0"
    created_timestamp: Optional[float] = None
    modified_timestamp: Optional[float] = None
    
    def validate(self) -> Dict[str, str]:
        """Validate configuration and return any errors."""
        errors = {}
        
        # Validate capture config
        if self.capture.max_file_size_kb <= 0:
            errors['capture.max_file_size_kb'] = "Must be greater than 0"
        
        if self.capture.max_file_size_kb > 50 * 1024:  # 50MB
            errors['capture.max_file_size_kb'] = "Exceeds reasonable limit (50MB)"
        
        if self.capture.capture_timeout_seconds <= 0:
            errors['capture.capture_timeout_seconds'] = "Must be greater than 0"
        
        # Validate service config
        if self.service.stream_interval_seconds < 0.1:
            errors['service.stream_interval_seconds'] = "Must be at least 0.1 seconds"
        
        if self.service.stream_interval_seconds > 300:  # 5 minutes
            errors['service.stream_interval_seconds'] = "Exceeds reasonable limit (300 seconds)"
        
        if self.service.max_archive_files <= 0:
            errors['service.max_archive_files'] = "Must be greater than 0"
        
        if self.service.max_archive_files > 10000:
            errors['service.max_archive_files'] = "Exceeds reasonable limit (10000 files)"
        
        if self.service.max_archive_age_days <= 0:
            errors['service.max_archive_age_days'] = "Must be greater than 0"
        
        if self.service.max_archive_age_days > 365:
            errors['service.max_archive_age_days'] = "Exceeds reasonable limit (365 days)"
        
        # Validate security config
        if self.security.max_session_age_hours <= 0:
            errors['security.max_session_age_hours'] = "Must be greater than 0"
        
        if self.security.max_session_age_hours > 24 * 30:  # 30 days
            errors['security.max_session_age_hours'] = "Exceeds reasonable limit (30 days)"
        
        # Validate storage paths if provided
        if self.storage.base_directory:
            try:
                base_path = Path(self.storage.base_directory)
                if not base_path.is_absolute():
                    errors['storage.base_directory'] = "Must be an absolute path"
            except Exception as e:
                errors['storage.base_directory'] = f"Invalid path: {e}"
        
        return errors
    
    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self.validate()) == 0


class VisionConfigManager:
    """Configuration manager for the vision system."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        if config_path:
            self.config_path = Path(config_path)
        else:
            # Default to secure location
            self.config_path = self._get_default_config_path()
        
        self.config: Optional[VisionSystemConfig] = None
        self.logger = logging.getLogger(__name__)
    
    def _get_default_config_path(self) -> Path:
        """Get default configuration file path."""
        if os.name == 'nt':
            # Windows: Use %LOCALAPPDATA%
            local_appdata = os.environ.get('LOCALAPPDATA')
            if local_appdata:
                config_dir = Path(local_appdata) / "AIVision"
            else:
                config_dir = Path.home() / "AppData" / "Local" / "AIVision"
        else:
            # Unix-like: Use XDG config directory
            config_dir = Path.home() / ".config" / "ai-vision"
        
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / "vision_config.json"
    
    def load_config(self) -> VisionSystemConfig:
        """Load configuration from file, creating defaults if needed."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                
                # Convert nested dictionaries back to dataclass instances
                capture_data = config_data.get('capture', {})
                service_data = config_data.get('service', {})
                security_data = config_data.get('security', {})
                storage_data = config_data.get('storage', {})
                
                self.config = VisionSystemConfig(
                    capture=CaptureConfig(**capture_data),
                    service=ServiceConfig(**service_data),
                    security=SecurityConfig(**security_data),
                    storage=StorageConfig(**storage_data),
                    config_version=config_data.get('config_version', '1.0'),
                    created_timestamp=config_data.get('created_timestamp'),
                    modified_timestamp=config_data.get('modified_timestamp')
                )
                
                # Validate loaded configuration
                errors = self.config.validate()
                if errors:
                    self.logger.warning(f"Configuration validation errors: {errors}")
                    # Continue with loaded config but log warnings
                
                self.logger.info(f"Configuration loaded from {self.config_path}")
                
            except Exception as e:
                self.logger.error(f"Failed to load configuration: {e}")
                self.logger.info("Creating default configuration")
                self.config = VisionSystemConfig()
                self.save_config()  # Save defaults
        else:
            self.logger.info("No configuration file found, creating defaults")
            self.config = VisionSystemConfig()
            self.save_config()
        
        return self.config
    
    def save_config(self) -> bool:
        """Save current configuration to file."""
        if not self.config:
            return False
        
        try:
            # Update timestamps
            import time
            current_time = time.time()
            
            if self.config.created_timestamp is None:
                self.config.created_timestamp = current_time
            self.config.modified_timestamp = current_time
            
            # Ensure parent directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert dataclass to dictionary with enum handling
            config_dict = self._config_to_dict(self.config)
            
            # Write to file atomically
            temp_path = self.config_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(config_dict, f, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            
            # Atomic rename
            os.replace(temp_path, self.config_path)
            
            self.logger.info(f"Configuration saved to {self.config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            return False
    
    def _config_to_dict(self, config: VisionSystemConfig) -> Dict[str, Any]:
        """Convert configuration to dictionary with proper enum serialization."""
        def convert_value(obj):
            if hasattr(obj, '__dict__'):
                # It's a dataclass or similar object
                result = {}
                for key, value in obj.__dict__.items():
                    result[key] = convert_value(value)
                return result
            elif isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, (list, tuple)):
                return [convert_value(item) for item in obj]
            else:
                return obj
        
        return convert_value(config)
    
    def update_capture_format(self, format_type: ImageFormat, quality: CompressionLevel):
        """Update capture format settings."""
        if not self.config:
            self.load_config()
        
        self.config.capture.default_format = format_type
        self.config.capture.default_quality = quality
        
        self.logger.info(f"Updated capture format: {format_type.value} @ {quality.value}")
        return self.save_config()
    
    def update_service_interval(self, interval_seconds: float):
        """Update service streaming interval."""
        if not self.config:
            self.load_config()
        
        if interval_seconds < 0.1:
            raise ValueError("Interval must be at least 0.1 seconds")
        
        self.config.service.stream_interval_seconds = interval_seconds
        
        self.logger.info(f"Updated service interval: {interval_seconds}s")
        return self.save_config()
    
    def update_max_file_size(self, size_kb: int):
        """Update maximum file size limit."""
        if not self.config:
            self.load_config()
        
        if size_kb <= 0:
            raise ValueError("File size must be greater than 0")
        
        self.config.capture.max_file_size_kb = size_kb
        
        self.logger.info(f"Updated max file size: {size_kb}KB")
        return self.save_config()
    
    def toggle_auto_optimization(self, enabled: bool):
        """Toggle automatic format optimization."""
        if not self.config:
            self.load_config()
        
        self.config.capture.auto_optimize_format = enabled
        
        self.logger.info(f"Auto optimization: {'enabled' if enabled else 'disabled'}")
        return self.save_config()
    
    def get_config(self) -> VisionSystemConfig:
        """Get current configuration, loading if necessary."""
        if not self.config:
            return self.load_config()
        return self.config
    
    def reset_to_defaults(self) -> bool:
        """Reset configuration to default values."""
        self.config = VisionSystemConfig()
        return self.save_config()
    
    def export_config(self, export_path: Union[str, Path]) -> bool:
        """Export configuration to a different file."""
        if not self.config:
            return False
        
        try:
            export_path = Path(export_path)
            config_dict = self._config_to_dict(self.config)
            
            with open(export_path, 'w') as f:
                json.dump(config_dict, f, indent=2, sort_keys=True)
            
            self.logger.info(f"Configuration exported to {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export configuration: {e}")
            return False


# Convenience function for global config access
_global_config_manager: Optional[VisionConfigManager] = None

def get_global_config() -> VisionSystemConfig:
    """Get the global configuration instance."""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = VisionConfigManager()
    return _global_config_manager.get_config()

def get_config_manager() -> VisionConfigManager:
    """Get the global configuration manager instance."""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = VisionConfigManager()
    return _global_config_manager


if __name__ == "__main__":
    # Test configuration system
    print("Testing Vision System Configuration...")
    
    config_manager = VisionConfigManager("test_config.json")
    
    # Load/create config
    config = config_manager.load_config()
    print(f"Loaded config: {config.config_version}")
    
    # Validate config
    errors = config.validate()
    if errors:
        print(f"Validation errors: {errors}")
    else:
        print("Configuration is valid")
    
    # Test updates
    config_manager.update_capture_format(ImageFormat.WEBP, CompressionLevel.MEDIUM)
    config_manager.update_service_interval(2.0)
    config_manager.update_max_file_size(1024)
    
    # Test save/reload
    config_manager.save_config()
    
    # Create new manager and load same config
    config_manager2 = VisionConfigManager("test_config.json")
    config2 = config_manager2.load_config()
    
    print(f"Reloaded format: {config2.capture.default_format.value}")
    print(f"Reloaded interval: {config2.service.stream_interval_seconds}")
    print(f"Reloaded max size: {config2.capture.max_file_size_kb}KB")
    
    print("Configuration system test completed!")