from typing import Optional, Type, Dict, Any, List
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import json


# ==================== Tool 1: 查询设备工具 ====================

class QueryDevicesInput(BaseModel):
    room_name: str = Field(description="The room name to query devices, e.g., 'living_room', 'master_bedroom'. Use 'all' to get all rooms.")


class QueryDevicesTool(BaseTool):
    name = "query_devices"
    description = """Query what devices are available in a specific room or all rooms. Input should be a JSON string with 'room_name' field. Example: {"room_name": "living_room"} or {"room_name": "all"}"""
    args_schema: Type[BaseModel] = QueryDevicesInput
    
    # Store current environment
    current_home_status: Dict[str, Any] = {}
    current_methods: List[Dict] = []
    
    def load_case(self, env_data: dict):
        """Load environment data for the current test case"""
        self.current_home_status = env_data.get('home_status', {})
        self.current_methods = env_data.get('method', [])
    
    def _run(self, room_name: str) -> str:
        """Query devices in a room or all rooms"""
        if not self.current_home_status:
            return "Error: Environment not loaded."
        
        try:
            # Handle JSON string input from LangChain
            if room_name.startswith('{'):
                import json
                parsed = json.loads(room_name)
                room_name = parsed.get('room_name', room_name)
            
            room_name = room_name.lower().strip()
            
            # Query all rooms
            if room_name == 'all':
                result = ["Available rooms and devices:"]
                for room, devices in self.current_home_status.items():
                    if isinstance(devices, dict):
                        device_list = [d for d in devices.keys() if d != 'room_name']
                        result.append(f"- {room}: {', '.join(device_list)}")
                return "\n".join(result)
            
            # Query specific room
            if room_name not in self.current_home_status:
                available_rooms = list(self.current_home_status.keys())
                return f"Room '{room_name}' not found. Available rooms: {', '.join(available_rooms)}"
            
            room_devices = self.current_home_status[room_name]
            result = [f"Devices in {room_name}:"]
            
            for device_name, device_info in room_devices.items():
                if device_name == 'room_name':
                    continue
                
                state = device_info.get('state', 'unknown')
                
                # Get available operations for this device
                operations = [
                    m['operation'] for m in self.current_methods
                    if m.get('room_name') == room_name and m.get('device_name') == device_name
                ]
                
                result.append(f"  - {device_name}: state={state}, operations={operations[:5]}")
            
            return "\n".join(result)
            
        except Exception as e:
            return f"Error querying devices: {str(e)}"
    
    def _arun(self, room_name: str):
        raise NotImplementedError("Async not implemented")


# ==================== Tool 2: 执行命令工具 ====================

class ExecuteCommandInput(BaseModel):
    command: str = Field(description="The command to execute in format: room_name.device_name.operation(parameters)")


class ExecuteCommandTool(BaseTool):
    name = "execute_command"
    description = """Execute a smart home command. Input should be a JSON string with 'command' field in format room_name.device_name.operation(parameters). Example: {"command": "living_room.light.turn_on()"}"""
    args_schema: Type[BaseModel] = ExecuteCommandInput
    
    # Store current environment
    current_home_status: Dict[str, Any] = {}
    current_methods: List[Dict] = []
    
    def load_case(self, env_data: dict):
        """Load environment data for the current test case"""
        self.current_home_status = env_data.get('home_status', {})
        self.current_methods = env_data.get('method', [])
    
    def _run(self, command: str) -> str:
        """Execute a smart home command with validation"""
        if not self.current_home_status:
            return "Error: Environment not loaded."
        
        try:
            # Handle JSON string input from LangChain
            if command.startswith('{'):
                import json
                parsed = json.loads(command)
                command = parsed.get('command', command)
            
            # Parse command: "room_name.device_name.operation(params)"
            clean_cmd = command.strip()
            
            # Split by '(' to separate operation and parameters
            if '(' in clean_cmd:
                cmd_part, param_part = clean_cmd.split('(', 1)
                param_part = param_part.rstrip(')')
            else:
                cmd_part = clean_cmd
                param_part = ""
            
            # Split command part by '.'
            parts = cmd_part.split('.')
            
            if len(parts) < 3:
                return f"Error: Invalid command format. Expected 'room_name.device_name.operation(parameters)'. Got: '{command}'"
            
            target_room = parts[0].lower().strip()
            target_device = parts[1].lower().strip()
            target_operation = parts[2].lower().strip()
            
            # Validation 1: Check room exists
            if target_room not in self.current_home_status:
                available_rooms = list(self.current_home_status.keys())
                return f"Error: Room '{target_room}' not found. Available rooms: {', '.join(available_rooms[:5])}. Use query_devices to check available rooms."
            
            # Validation 2: Check device exists in room
            room_devices = self.current_home_status[target_room]
            if target_device not in room_devices:
                available_devices = [d for d in room_devices.keys() if d != 'room_name']
                return f"Error: Device '{target_device}' not found in '{target_room}'. Available devices: {', '.join(available_devices)}. Use query_devices(room_name='{target_room}') to check."
            
            # Validation 3: Check operation is valid
            valid_operation = False
            for method in self.current_methods:
                if (method.get('room_name') == target_room and 
                    method.get('device_name') == target_device and 
                    method.get('operation') == target_operation):
                    valid_operation = True
                    break
            
            if not valid_operation:
                available_ops = [
                    m['operation'] for m in self.current_methods 
                    if m.get('room_name') == target_room and m.get('device_name') == target_device
                ]
                return f"Error: Operation '{target_operation}' not supported for '{target_room}.{target_device}'. Available operations: {', '.join(available_ops[:5])}"
            
            # All validations passed - execute command and update state
            self._update_device_state(target_room, target_device, target_operation, param_part)
            return f"Success: Executed {target_room}.{target_device}.{target_operation}({param_part})"
            
        except Exception as e:
            return f"Error: Command execution failed. {str(e)}"
    
    def _update_device_state(self, room: str, device: str, operation: str, params: str):
        """Update device state after command execution"""
        try:
            device_info = self.current_home_status[room][device]
            
            # Handle different operation types
            if operation == "turn_on":
                device_info["state"] = "on"
            
            elif operation == "turn_off":
                device_info["state"] = "off"
            
            elif operation == "open":
                device_info["state"] = "open"
            
            elif operation == "close":
                device_info["state"] = "closed"
            
            elif operation.startswith("set_"):
                # Extract attribute name and value
                attr_name = operation.replace("set_", "")
                
                # Parse parameter value
                if params:
                    try:
                        # Try to convert to number
                        if '.' in params:
                            value = float(params)
                        else:
                            value = int(params)
                    except ValueError:
                        value = params.strip()
                    
                    # Update attribute
                    if "attributes" not in device_info:
                        device_info["attributes"] = {}
                    
                    # Store as dict with value key (matching HomeBench format)
                    device_info["attributes"][attr_name] = {"value": value}
                    
                    # Also update state for certain attributes
                    if attr_name in ["brightness", "temperature", "intensity", "degree"]:
                        device_info["state"] = "on"  # Assume device is on when setting values
            
            elif operation.startswith("increase_") or operation.startswith("decrease_"):
                # Handle relative changes
                action = "increase" if operation.startswith("increase_") else "decrease"
                attr_name = operation.replace(f"{action}_", "")
                
                if params:
                    try:
                        change = float(params) if '.' in params else int(params)
                        
                        if "attributes" not in device_info:
                            device_info["attributes"] = {}
                        
                        # Get current value
                        current = device_info["attributes"].get(attr_name, {"value": 0})
                        if isinstance(current, dict):
                            current_value = current.get("value", 0)
                        else:
                            current_value = current
                        
                        # Calculate new value
                        if action == "increase":
                            new_value = current_value + change
                        else:
                            new_value = current_value - change
                        
                        # Update
                        device_info["attributes"][attr_name] = {"value": new_value}
                        device_info["state"] = "on"
                    except (ValueError, TypeError):
                        pass  # Keep current state if parsing fails
            
        except Exception as e:
            # If state update fails, just log it but don't fail the command
            # This ensures backward compatibility
            pass
    
    def _arun(self, command: str):
        raise NotImplementedError("Async not implemented")


# ==================== Tool 3: 获取设备状态工具 ====================

class GetDeviceStateInput(BaseModel):
    room_name: str = Field(description="The room name")
    device_name: str = Field(description="The device name")


class GetDeviceStateTool(BaseTool):
    name = "get_device_state"
    description = """Get the current state of a specific device. Input should be a JSON string with 'room_name' and 'device_name' fields. Example: {"room_name": "living_room", "device_name": "light"}"""
    args_schema: Type[BaseModel] = GetDeviceStateInput
    
    current_home_status: Dict[str, Any] = {}
    
    def load_case(self, env_data: dict):
        """Load environment data"""
        self.current_home_status = env_data.get('home_status', {})
    
    def _run(self, room_name: str = None, device_name: str = None, **kwargs) -> str:
        """Get device state"""
        if not self.current_home_status:
            return "Error: Environment not loaded."
        
        try:
            # Handle JSON string input from LangChain (when passed as single argument)
            if room_name and isinstance(room_name, str) and room_name.startswith('{'):
                import json
                parsed = json.loads(room_name)
                room_name = parsed.get('room_name')
                device_name = parsed.get('device_name')
            
            # Handle dict input
            if isinstance(room_name, dict):
                device_name = room_name.get('device_name')
                room_name = room_name.get('room_name')
            
            if not room_name or not device_name:
                return "Error: Both room_name and device_name are required."
            
            room_name = room_name.lower().strip()
            device_name = device_name.lower().strip()
            
            if room_name not in self.current_home_status:
                return f"Error: Room '{room_name}' not found."
            
            room_devices = self.current_home_status[room_name]
            if device_name not in room_devices:
                return f"Error: Device '{device_name}' not found in '{room_name}'."
            
            device_info = room_devices[device_name]
            state = device_info.get('state', 'unknown')
            attributes = device_info.get('attributes', {})
            
            result = [f"Device: {room_name}.{device_name}"]
            result.append(f"State: {state}")
            
            if attributes:
                result.append("Attributes:")
                for attr_name, attr_value in attributes.items():
                    if isinstance(attr_value, dict):
                        value = attr_value.get('value', 'N/A')
                        result.append(f"  - {attr_name}: {value}")
                    else:
                        result.append(f"  - {attr_name}: {attr_value}")
            
            return "\n".join(result)
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _arun(self, room_name: str, device_name: str):
        raise NotImplementedError("Async not implemented")
