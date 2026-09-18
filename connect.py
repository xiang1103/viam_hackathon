import os
from dotenv import load_dotenv
from viam.robot.client import RobotClient
load_dotenv()
api_key = os.getenv("API_KEY")
api_key_id = os.getenv("API_KEY_ID")
robot_address = os.getenv("ROBOT_ADDRESS")

# create connection 
async def connect() -> RobotClient:
    opts = RobotClient.Options.with_api_key(
        api_key=api_key,
        api_key_id=api_key_id
    )
    return await RobotClient.at_address(robot_address, opts)