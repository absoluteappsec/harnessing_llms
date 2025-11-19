import re
from agentic_basic import analyze_code
import pytest

@pytest.mark.integration
def test_analyze_code_insecure():
    """
    Test if the agent identifies insecure code.
    """
    input_code = """
    @login_required
    @user_passes_test(can_create_project)
    def update_user_active(request):
        user_id = request.GET.get('user_id')
        User.objects.filter(id=user_id).update(is_active=False)
    """
    response = analyze_code(input_code)
    response = str(response)

    # Use regex to assert "is_insecure" is true or True
    assert re.search(r'"is_insecure"\s*:\s*true', response, re.IGNORECASE), \
        "The code should be marked as insecure."

