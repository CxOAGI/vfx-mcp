"""MCP protocol verification tests for VFX MCP server.

This module provides comprehensive tests for Model Context Protocol (MCP)
compliance and protocol-specific functionality. Tests ensure the server
correctly implements MCP specification requirements.

These tests use the fastmcp 2.9.0 high-level Client API. The client is always
used as an async context manager (``async with Client(mcp_server) as client:``),
which performs the MCP initialize handshake automatically. Handshake details
are inspected via ``client.initialize_result``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

if TYPE_CHECKING:
    from pathlib import Path

    from fastmcp import FastMCP


class TestMCPProtocolVerification:
    """Test suite for MCP protocol compliance verification."""

    @pytest.mark.integration
    async def test_mcp_jsonrpc_compliance(self, mcp_server: FastMCP[None]) -> None:
        """Test the MCP initialize handshake exposes required fields."""
        async with Client(mcp_server) as client:
            init_result = client.initialize_result

            # Initialize result should expose the core handshake fields.
            assert init_result.protocolVersion is not None
            assert init_result.capabilities is not None
            assert init_result.serverInfo is not None

            # Protocol version should be a valid dated version string.
            protocol_version = init_result.protocolVersion
            assert isinstance(protocol_version, str)
            assert protocol_version.startswith("202")

    @pytest.mark.integration
    async def test_mcp_capabilities_declaration(
        self,
        mcp_server: FastMCP[None],
    ) -> None:
        """Test MCP capabilities are properly declared."""
        async with Client(mcp_server) as client:
            capabilities = client.initialize_result.capabilities

            # Server should declare tool capabilities.
            assert capabilities.tools is not None

            # Server should declare resource capabilities.
            assert capabilities.resources is not None

    @pytest.mark.integration
    async def test_mcp_server_info(self, mcp_server: FastMCP[None]) -> None:
        """Test MCP server information is properly provided."""
        async with Client(mcp_server) as client:
            server_info = client.initialize_result.serverInfo

            # Verify server identity.
            assert server_info.name == "vfx-mcp"
            assert isinstance(server_info.version, str)

    @pytest.mark.integration
    async def test_mcp_tools_list_compliance(self, mcp_server: FastMCP[None]) -> None:
        """Test MCP tools/list request compliance."""
        async with Client(mcp_server) as client:
            tools = await client.list_tools()

            assert isinstance(tools, list)
            assert len(tools) > 0

            # Each tool should have required MCP fields.
            for tool in tools:
                assert isinstance(tool.name, str)
                assert isinstance(tool.description, str)
                assert isinstance(tool.inputSchema, dict)

                # inputSchema should be valid JSON Schema.
                schema = tool.inputSchema
                assert "type" in schema
                assert schema["type"] == "object"

                if "properties" in schema:
                    assert isinstance(schema["properties"], dict)

                if "required" in schema:
                    assert isinstance(schema["required"], list)

    @pytest.mark.integration
    async def test_mcp_resources_list_compliance(
        self,
        mcp_server: FastMCP[None],
    ) -> None:
        """Test MCP resources/list request compliance."""
        async with Client(mcp_server) as client:
            resources = await client.list_resources()

            assert isinstance(resources, list)
            assert len(resources) > 0

            # Each resource should have required MCP fields.
            for resource in resources:
                uri = str(resource.uri)
                assert isinstance(uri, str)
                assert isinstance(resource.name, str)

                # URI should be valid.
                assert uri.startswith("videos://")

    @pytest.mark.integration
    async def test_mcp_tools_call_compliance(
        self,
        sample_video: Path,
        mcp_server: FastMCP[None],
    ) -> None:
        """Test MCP tools/call request compliance."""
        async with Client(mcp_server) as client:
            content = await client.call_tool(
                "get_video_info",
                {"video_path": str(sample_video)},
            )

            # Result should be a non-empty list of content blocks.
            assert isinstance(content, list)
            assert len(content) > 0

            for block in content:
                assert block.type == "text"
                assert isinstance(block.text, str)

    @pytest.mark.integration
    async def test_mcp_resources_read_compliance(
        self,
        mcp_server: FastMCP[None],
    ) -> None:
        """Test MCP resources/read request compliance."""
        async with Client(mcp_server) as client:
            content = await client.read_resource("videos://list")

            # Result should be a non-empty list of content blocks.
            assert isinstance(content, list)
            assert len(content) > 0

            for block in content:
                assert isinstance(block.text, str)

    @pytest.mark.integration
    async def test_mcp_error_handling_compliance(
        self,
        mcp_server: FastMCP[None],
    ) -> None:
        """Test MCP error handling compliance."""
        async with Client(mcp_server) as client:
            # An invalid tool call surfaces as a ToolError on the client.
            with pytest.raises(ToolError):
                await client.call_tool(
                    "nonexistent_tool",
                    {"invalid": "parameters"},
                )

    @pytest.mark.integration
    async def test_mcp_tool_schema_validation(self, mcp_server: FastMCP[None]) -> None:
        """Test MCP tool input schema validation."""
        async with Client(mcp_server) as client:
            tools = await client.list_tools()

            # Find trim_video tool to test schema validation.
            trim_tool = None
            for tool in tools:
                if tool.name == "trim_video":
                    trim_tool = tool
                    break

            assert trim_tool is not None, "trim_video tool not found"

            # Test schema structure.
            schema = trim_tool.inputSchema
            assert "type" in schema
            assert schema["type"] == "object"
            assert "properties" in schema
            assert "required" in schema

            # Verify properties.
            properties = schema["properties"]
            required = schema["required"]

            assert "input_path" in properties
            assert "output_path" in properties
            assert "start_time" in properties
            assert "duration" in properties

            # input_path, output_path, and start_time are required;
            # duration is now optional (trims to end when omitted).
            assert "input_path" in required
            assert "output_path" in required
            assert "start_time" in required

    @pytest.mark.integration
    async def test_mcp_resource_uri_compliance(self, mcp_server: FastMCP[None]) -> None:
        """Test MCP resource URI scheme compliance."""
        async with Client(mcp_server) as client:
            resources = await client.list_resources()

            # Test URI format compliance.
            for resource in resources:
                uri = str(resource.uri)

                # URI should follow videos:// scheme.
                assert uri.startswith("videos://")

                # Each resource should carry a non-empty name.
                assert isinstance(resource.name, str)
                assert len(resource.name) > 0

    @pytest.mark.integration
    async def test_mcp_content_types(self, mcp_server: FastMCP[None]) -> None:
        """Test MCP content type handling."""
        async with Client(mcp_server) as client:
            content = await client.read_resource("videos://list")

            for block in content:
                assert isinstance(block.text, str)

                # Content should be valid JSON for the list endpoint.
                try:
                    json.loads(block.text)
                except json.JSONDecodeError:
                    pytest.fail("Resource content is not valid JSON")

    @pytest.mark.integration
    async def test_mcp_progress_reporting(
        self,
        sample_video: Path,
        temp_dir: Path,
        mcp_server: FastMCP[None],
    ) -> None:
        """Test MCP progress reporting capabilities."""
        async with Client(mcp_server) as client:
            # A successful tool call should complete without raising.
            output_path = temp_dir / "progress_test.mp4"
            _ = await client.call_tool(
                "trim_video",
                {
                    "input_path": str(sample_video),
                    "output_path": str(output_path),
                    "start_time": 0,
                    "duration": 2,
                },
            )

            # Output file should exist.
            assert output_path.exists()

    @pytest.mark.integration
    async def test_mcp_concurrent_requests(self, mcp_server: FastMCP[None]) -> None:
        """Test MCP concurrent request handling."""
        import asyncio

        async with Client(mcp_server) as client:
            # Create multiple concurrent requests.
            async def make_request() -> list:
                return await client.list_tools()

            tasks = [make_request() for _ in range(5)]
            results = await asyncio.gather(*tasks)

            # All requests should succeed.
            assert len(results) == 5

            # All results should list the same tools.
            first_names = sorted(t.name for t in results[0])
            for result in results[1:]:
                assert sorted(t.name for t in result) == first_names

    @pytest.mark.integration
    async def test_mcp_request_isolation(
        self,
        sample_video: Path,
        temp_dir: Path,
        mcp_server: FastMCP[None],
    ) -> None:
        """Test MCP request isolation and state management."""
        async with Client(mcp_server) as client:
            # Make multiple tool calls.
            for i in range(3):
                output_path = temp_dir / f"isolation_test_{i}.mp4"
                _ = await client.call_tool(
                    "trim_video",
                    {
                        "input_path": str(sample_video),
                        "output_path": str(output_path),
                        "start_time": 0,
                        "duration": 1,
                    },
                )

            # All output files should exist independently.
            for i in range(3):
                output_path = temp_dir / f"isolation_test_{i}.mp4"
                assert output_path.exists()
