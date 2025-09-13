// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title MaaSRegistry (Simplified)
/// @notice Minimal identity registry for commuters and providers
contract MaaSRegistry {
    address public owner;

    // IDs are arbitrary off-chain identifiers decided by the marketplace.
    mapping(uint256 => address) public commuterAddressById;
    mapping(uint256 => address) public providerAddressById;
    mapping(uint256 => uint8)   public providerModeById; // e.g., 1 = ridehail, 2 = bus, etc.

    event CommuterRegistered(uint256 indexed commuterId, address indexed account);
    event ProviderRegistered(uint256 indexed providerId, address indexed account, uint8 mode);

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "zero");
        owner = newOwner;
    }

    function registerCommuter(uint256 commuterId, address account) external onlyOwner {
        require(account != address(0), "zero");
        require(commuterAddressById[commuterId] == address(0), "exists");
        commuterAddressById[commuterId] = account;
        emit CommuterRegistered(commuterId, account);
    }

    function registerProvider(uint256 providerId, address account, uint8 mode) external onlyOwner {
        require(account != address(0), "zero");
        require(providerAddressById[providerId] == address(0), "exists");
        providerAddressById[providerId] = account;
        providerModeById[providerId]   = mode;
        emit ProviderRegistered(providerId, account, mode);
    }

    // Convenience getters
    function getCommuter(uint256 commuterId) external view returns (address) {
        return commuterAddressById[commuterId];
    }

    function getProvider(uint256 providerId) external view returns (address account, uint8 mode) {
        return (providerAddressById[providerId], providerModeById[providerId]);
    }
}
