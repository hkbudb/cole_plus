pragma solidity >=0.4.0 <0.7.0;

contract ETH {
    mapping(string => uint256) account;

    function operate(string memory arg0, string memory arg1) public {
        account[arg0] += 1;
        account[arg1] += 1;
    }

    function get_value(string memory arg0) public view returns (uint256 balance) {
        return account[arg0];
    }
}