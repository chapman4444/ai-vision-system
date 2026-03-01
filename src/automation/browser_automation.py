#!/usr/bin/env python3
"""
Browser-Specific Automation
Enhanced automation for web browsers with special handling for:
- Web form interactions
- JavaScript-heavy pages
- Browser-specific elements
- Cross-browser compatibility
"""

import time
from typing import List, Dict, Optional, Tuple
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import pyautogui
from .universal_input import UniversalInputController
from .gui_elements import GUIAutomator, GUIElement

class BrowserAutomator:
    """
    Enhanced browser automation combining Selenium and vision-based approaches
    Fallback to vision-based automation when Selenium fails
    """
    
    def __init__(self, browser: str = "chrome", headless: bool = False, 
                 vision_fallback: bool = True):
        self.browser = browser.lower()
        self.headless = headless
        self.vision_fallback = vision_fallback
        self.driver = None
        
        # Vision-based fallback
        if vision_fallback:
            self.gui_automator = GUIAutomator()
            self.input_controller = self.gui_automator.input_controller
        
        # Browser-specific configurations
        self.browser_configs = {
            "chrome": self._setup_chrome,
            "firefox": self._setup_firefox,
            "edge": self._setup_edge
        }
        
    def start_browser(self, url: str = None) -> bool:
        """Start browser with optional initial URL"""
        try:
            if self.browser in self.browser_configs:
                self.driver = self.browser_configs[self.browser]()
            else:
                raise ValueError(f"Unsupported browser: {self.browser}")
                
            if url:
                self.driver.get(url)
                
            return True
            
        except Exception as e:
            print(f"Failed to start browser: {e}")
            return False
            
    def _setup_chrome(self):
        """Setup Chrome WebDriver"""
        options = ChromeOptions()
        if self.headless:
            options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        return webdriver.Chrome(options=options)
        
    def _setup_firefox(self):
        """Setup Firefox WebDriver"""
        options = FirefoxOptions()
        if self.headless:
            options.add_argument("--headless")
        options.set_preference("dom.webdriver.enabled", False)
        
        return webdriver.Firefox(options=options)
        
    def _setup_edge(self):
        """Setup Edge WebDriver"""
        options = webdriver.EdgeOptions()
        if self.headless:
            options.add_argument("--headless")
        options.add_argument("--disable-blink-features=AutomationControlled")
        
        return webdriver.Edge(options=options)
        
    def navigate_to(self, url: str) -> bool:
        """Navigate to URL"""
        try:
            if self.driver:
                self.driver.get(url)
                return True
            return False
        except Exception as e:
            print(f"Navigation failed: {e}")
            return False
            
    def click_element(self, selector: str = None, text: str = None, 
                     timeout: float = 10.0, vision_fallback: bool = True) -> bool:
        """
        Click element by CSS selector or text content
        Falls back to vision-based clicking if Selenium fails
        """
        # Try Selenium first
        if self.driver and (selector or text):
            try:
                wait = WebDriverWait(self.driver, timeout)
                
                if selector:
                    element = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))
                else:
                    # Find by text content
                    element = wait.until(EC.element_to_be_clickable(
                        (By.XPATH, f"//*[contains(text(), '{text}')]")
                    ))
                
                # Scroll element into view
                self.driver.execute_script("arguments[0].scrollIntoView(true);", element)
                time.sleep(0.5)
                
                element.click()
                return True
                
            except (TimeoutException, NoSuchElementException) as e:
                print(f"Selenium click failed: {e}")
                
        # Vision-based fallback
        if self.vision_fallback and text:
            return self.gui_automator.click_button_by_text(text, timeout)
            
        return False
        
    def fill_form_field(self, selector: str = None, field_name: str = None, 
                       value: str = "", clear_first: bool = True, 
                       vision_fallback: bool = True) -> bool:
        """
        Fill form field by selector, name, or using vision
        """
        # Try Selenium first
        if self.driver and (selector or field_name):
            try:
                wait = WebDriverWait(self.driver, 10)
                
                if selector:
                    element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                else:
                    element = wait.until(EC.presence_of_element_located((By.NAME, field_name)))
                
                # Scroll into view and focus
                self.driver.execute_script("arguments[0].scrollIntoView(true);", element)
                element.click()
                
                if clear_first:
                    element.clear()
                    
                element.send_keys(value)
                return True
                
            except (TimeoutException, NoSuchElementException) as e:
                print(f"Selenium form fill failed: {e}")
                
        # Vision-based fallback
        if self.vision_fallback:
            return self.gui_automator.fill_text_field(value, clear_first=clear_first)
            
        return False
        
    def select_dropdown_option(self, selector: str = None, option_text: str = "",
                             option_value: str = "", vision_fallback: bool = True) -> bool:
        """
        Select dropdown option by text or value
        """
        # Try Selenium first
        if self.driver and selector:
            try:
                wait = WebDriverWait(self.driver, 10)
                dropdown_element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                
                select = Select(dropdown_element)
                
                if option_text:
                    select.select_by_visible_text(option_text)
                elif option_value:
                    select.select_by_value(option_value)
                else:
                    return False
                    
                return True
                
            except (TimeoutException, NoSuchElementException) as e:
                print(f"Selenium dropdown selection failed: {e}")
                
        # Vision-based fallback
        if self.vision_fallback and option_text:
            return self.gui_automator.select_dropdown_option(option_text)
            
        return False
        
    def wait_for_element(self, selector: str, timeout: float = 10.0) -> bool:
        """Wait for element to be present"""
        if not self.driver:
            return False
            
        try:
            wait = WebDriverWait(self.driver, timeout)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
            return True
        except TimeoutException:
            return False
            
    def wait_for_page_load(self, timeout: float = 30.0) -> bool:
        """Wait for page to fully load"""
        if not self.driver:
            return False
            
        try:
            wait = WebDriverWait(self.driver, timeout)
            wait.until(lambda driver: driver.execute_script("return document.readyState") == "complete")
            return True
        except TimeoutException:
            return False
            
    def execute_javascript(self, script: str) -> any:
        """Execute JavaScript in the browser"""
        if not self.driver:
            return None
            
        try:
            return self.driver.execute_script(script)
        except Exception as e:
            print(f"JavaScript execution failed: {e}")
            return None
            
    def take_screenshot(self, filename: str = None) -> str:
        """Take screenshot of current page"""
        if not self.driver:
            return None
            
        try:
            if not filename:
                timestamp = int(time.time())
                filename = f"browser_screenshot_{timestamp}.png"
                
            filepath = self.driver.save_screenshot(filename)
            return filename if filepath else None
        except Exception as e:
            print(f"Screenshot failed: {e}")
            return None
            
    def scroll_page(self, direction: str = "down", amount: int = 3) -> bool:
        """Scroll page using keyboard or JavaScript"""
        try:
            if direction.lower() == "down":
                for _ in range(amount):
                    self.input_controller.key_press("page_down")
                    time.sleep(0.1)
            elif direction.lower() == "up":
                for _ in range(amount):
                    self.input_controller.key_press("page_up") 
                    time.sleep(0.1)
            elif direction.lower() == "bottom":
                self.input_controller.key_combination(["ctrl", "end"])
            elif direction.lower() == "top":
                self.input_controller.key_combination(["ctrl", "home"])
                
            return True
        except Exception as e:
            print(f"Scroll failed: {e}")
            return False
            
    def handle_alert(self, action: str = "accept") -> bool:
        """Handle JavaScript alerts, confirms, prompts"""
        if not self.driver:
            return False
            
        try:
            alert = self.driver.switch_to.alert
            
            if action.lower() == "accept":
                alert.accept()
            elif action.lower() == "dismiss":
                alert.dismiss()
            else:
                # For prompts, send text then accept
                alert.send_keys(action)
                alert.accept()
                
            return True
        except Exception as e:
            print(f"Alert handling failed: {e}")
            return False
            
    def switch_tab(self, tab_index: int = -1) -> bool:
        """Switch to different browser tab"""
        if not self.driver:
            return False
            
        try:
            handles = self.driver.window_handles
            if tab_index == -1:  # Switch to last tab
                tab_index = len(handles) - 1
                
            if 0 <= tab_index < len(handles):
                self.driver.switch_to.window(handles[tab_index])
                return True
            return False
        except Exception as e:
            print(f"Tab switch failed: {e}")
            return False
            
    def open_new_tab(self, url: str = None) -> bool:
        """Open new browser tab"""
        if not self.driver:
            return False
            
        try:
            self.driver.execute_script("window.open('', '_blank');")
            self.switch_tab(-1)  # Switch to new tab
            
            if url:
                self.driver.get(url)
                
            return True
        except Exception as e:
            print(f"New tab failed: {e}")
            return False
            
    def close_current_tab(self) -> bool:
        """Close current browser tab"""
        if not self.driver:
            return False
            
        try:
            handles = self.driver.window_handles
            if len(handles) > 1:
                self.driver.close()
                self.switch_tab(0)  # Switch to first remaining tab
            else:
                self.driver.close()
                self.driver = None
            return True
        except Exception as e:
            print(f"Close tab failed: {e}")
            return False
            
    def get_page_info(self) -> Dict[str, str]:
        """Get current page information"""
        if not self.driver:
            return {}
            
        try:
            return {
                "url": self.driver.current_url,
                "title": self.driver.title,
                "page_source_length": len(self.driver.page_source)
            }
        except Exception as e:
            print(f"Get page info failed: {e}")
            return {}
            
    def find_elements_by_text(self, text: str, partial: bool = True) -> List[Dict[str, str]]:
        """Find all elements containing specific text"""
        if not self.driver:
            return []
            
        try:
            if partial:
                xpath = f"//*[contains(text(), '{text}')]"
            else:
                xpath = f"//*[text()='{text}']"
                
            elements = self.driver.find_elements(By.XPATH, xpath)
            
            result = []
            for elem in elements:
                try:
                    result.append({
                        "tag": elem.tag_name,
                        "text": elem.text,
                        "location": str(elem.location),
                        "size": str(elem.size)
                    })
                except Exception:
                    continue
                    
            return result
        except Exception as e:
            print(f"Find elements failed: {e}")
            return []
            
    def close_browser(self):
        """Close browser and cleanup"""
        if self.driver:
            try:
                self.driver.quit()
            except Exception as e:
                print(f"Browser close failed: {e}")
            finally:
                self.driver = None

# High-level browser automation interface
class WebAutomationSuite:
    """Complete web automation suite combining all browser capabilities"""
    
    def __init__(self, browser: str = "chrome"):
        self.browser_automator = BrowserAutomator(browser=browser, vision_fallback=True)
        
    def automate_form_filling(self, url: str, form_data: Dict[str, str]) -> bool:
        """Automate complete form filling process"""
        if not self.browser_automator.start_browser(url):
            return False
            
        # Wait for page load
        self.browser_automator.wait_for_page_load()
        
        success = True
        for field_name, value in form_data.items():
            if not self.browser_automator.fill_form_field(field_name=field_name, value=value):
                print(f"Failed to fill field: {field_name}")
                success = False
                
        return success
        
    def automate_search_and_click(self, search_url: str, search_term: str, 
                                 result_text: str) -> bool:
        """Automate search and click on specific result"""
        if not self.browser_automator.start_browser(search_url):
            return False
            
        # Fill search field (try common search field selectors)
        search_selectors = ["input[name='q']", "input[name='search']", "#search", ".search-input"]
        search_filled = False
        
        for selector in search_selectors:
            if self.browser_automator.fill_form_field(selector=selector, value=search_term):
                search_filled = True
                break
                
        if not search_filled:
            # Vision fallback
            if not self.browser_automator.gui_automator.fill_text_field(search_term):
                return False
                
        # Submit search (try Enter key)
        self.browser_automator.input_controller.key_press("enter")
        
        # Wait for results and click on specific result
        time.sleep(2)
        return self.browser_automator.click_element(text=result_text)
        
    def cleanup(self):
        """Cleanup resources"""
        self.browser_automator.close_browser()